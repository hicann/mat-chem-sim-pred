/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @author Liu Fei (@Magic_LF)
 */

/*!
 * \file test_lj_force.cpp
 * \brief LJForce 算子完整测试 (精度 + 性能)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include "acl/acl.h"

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

extern "C" int32_t aclnnLJForceDirect(
    void* positionsAddr,
    void* forcesAddr,
    void* energyAddr,
    int32_t numAtoms,
    float epsilon,
    float sigma,
    float cutoff,
    void* workspaceAddr,
    uint64_t workspaceSize,
    void* stream
);

extern "C" uint64_t aclnnLJForceGetWorkspaceSize(int32_t numAtoms);

// CPU 参考实现
void ljForceCPU(const std::vector<float>& positions, int32_t numAtoms,
                float epsilon, float sigma, float cutoff,
                std::vector<float>& forces, float& energy) {
    forces.assign(numAtoms * 3, 0.0f);
    energy = 0.0f;

    float cutoffSq = cutoff * cutoff;
    float sigma2 = sigma * sigma;
    float sigma6 = sigma2 * sigma2 * sigma2;
    float eps4 = 4.0f * epsilon;
    float eps24 = 24.0f * epsilon;

    for (int32_t i = 0; i < numAtoms; i++) {
        for (int32_t j = i + 1; j < numAtoms; j++) {
            float dx = positions[i * 3] - positions[j * 3];
            float dy = positions[i * 3 + 1] - positions[j * 3 + 1];
            float dz = positions[i * 3 + 2] - positions[j * 3 + 2];
            float r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < cutoffSq && r2 > 1e-10f) {
                float r2inv = 1.0f / r2;
                float r6inv = r2inv * r2inv * r2inv;
                float s6r6 = sigma6 * r6inv;
                float s12r12 = s6r6 * s6r6;

                energy += eps4 * (s12r12 - s6r6);

                float fscalar = eps24 * r2inv * (2.0f * s12r12 - s6r6);
                forces[i * 3] += fscalar * dx;
                forces[i * 3 + 1] += fscalar * dy;
                forces[i * 3 + 2] += fscalar * dz;
                forces[j * 3] -= fscalar * dx;
                forces[j * 3 + 1] -= fscalar * dy;
                forces[j * 3 + 2] -= fscalar * dz;
            }
        }
    }
}

class LJForceNPU {
public:
    LJForceNPU() : initialized_(false), stream_(nullptr) {}

    ~LJForceNPU() {
        if (initialized_) {
            aclrtDestroyStream(stream_);
            aclrtResetDevice(0);
            aclFinalize();
        }
    }

    bool init() {
        if (initialized_) return true;
        auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != 100002) return false;
        ret = aclrtSetDevice(0);
        if (ret != ACL_SUCCESS) return false;
        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) return false;
        initialized_ = true;
        return true;
    }

    bool compute(const std::vector<float>& positions, int32_t numAtoms,
                 float epsilon, float sigma, float cutoff,
                 std::vector<float>& forces, float& energy) {
        if (!initialized_ && !init()) return false;

        // 计算核数和对齐
        int32_t maxCores = 32, minAtomsPerCore = 16;
        int32_t optimalCores = std::min((numAtoms + minAtomsPerCore - 1) / minAtomsPerCore, maxCores);
        optimalCores = std::max(optimalCores, 1);
        int32_t atomsPerCore = (numAtoms + optimalCores - 1) / optimalCores;
        int32_t forceStride = ((atomsPerCore * 3 + 7) / 8) * 8;
        int32_t forceTotalSize = forceStride * optimalCores;

        // 分配内存
        size_t posSize = numAtoms * 3 * sizeof(float);
        size_t forceSize = forceTotalSize * sizeof(float);
        size_t energySize = 8 * optimalCores * sizeof(float);
        uint64_t workspaceSize = aclnnLJForceGetWorkspaceSize(numAtoms);

        void *posAddr, *forcesAddr, *energyAddr, *workspaceAddr;
        aclrtMalloc(&posAddr, posSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&forcesAddr, forceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&energyAddr, energySize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        // 拷贝输入
        aclrtMemcpy(posAddr, posSize, positions.data(), posSize, ACL_MEMCPY_HOST_TO_DEVICE);

        // 调用算子
        aclnnLJForceDirect(posAddr, forcesAddr, energyAddr, numAtoms,
                           epsilon, sigma, cutoff, workspaceAddr, workspaceSize, stream_);
        aclrtSynchronizeStream(stream_);

        // 拷贝输出
        std::vector<float> forcesAligned(forceTotalSize, 0.0f);
        std::vector<float> energyBuf(8 * optimalCores, 0.0f);
        aclrtMemcpy(forcesAligned.data(), forceSize, forcesAddr, forceSize, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(energyBuf.data(), energySize, energyAddr, energySize, ACL_MEMCPY_DEVICE_TO_HOST);

        // 释放内存
        aclrtFree(posAddr);
        aclrtFree(forcesAddr);
        aclrtFree(energyAddr);
        aclrtFree(workspaceAddr);

        // 提取力数据
        forces.assign(numAtoms * 3, 0.0f);
        for (int32_t coreIdx = 0; coreIdx < optimalCores; coreIdx++) {
            int32_t startAtom = coreIdx * atomsPerCore;
            int32_t endAtom = std::min(startAtom + atomsPerCore, numAtoms);
            int32_t srcOffset = coreIdx * forceStride;
            for (int32_t localIdx = 0; localIdx < endAtom - startAtom; localIdx++) {
                int32_t atomIdx = startAtom + localIdx;
                forces[atomIdx * 3] = forcesAligned[srcOffset + localIdx * 3];
                forces[atomIdx * 3 + 1] = forcesAligned[srcOffset + localIdx * 3 + 1];
                forces[atomIdx * 3 + 2] = forcesAligned[srcOffset + localIdx * 3 + 2];
            }
        }

        // 累加能量
        energy = 0.0f;
        for (int32_t coreIdx = 0; coreIdx < optimalCores; coreIdx++) {
            energy += energyBuf[coreIdx * 8];
        }

        return true;
    }

    aclrtStream getStream() { return stream_; }

private:
    bool initialized_;
    aclrtStream stream_;
};

// 生成随机坐标
void generatePositions(std::vector<float>& positions, int32_t numAtoms, float boxSize, uint32_t seed) {
    srand(seed);
    positions.resize(numAtoms * 3);
    for (int32_t i = 0; i < numAtoms * 3; i++) {
        positions[i] = (float)rand() / RAND_MAX * boxSize;
    }
}

void testCPU() {
    LOG_PRINT("============================================================\n");
    LOG_PRINT("CPU 参考实现测试\n");
    LOG_PRINT("============================================================\n");

    float sigma = 3.4f, epsilon = 0.01f, cutoff = 10.0f;

    LOG_PRINT("\n--- 两原子测试 ---\n");
    std::vector<float> positions = {0, 0, 0, sigma, 0, 0};
    std::vector<float> forces;
    float energy;
    ljForceCPU(positions, 2, epsilon, sigma, cutoff, forces, energy);
    float totalForce = forces[0] + forces[3];
    const char* status = (std::abs(totalForce) < 1e-6f) ? "PASS" : "FAIL";
    LOG_PRINT("  能量: %.6f, 总力: %.6f, 牛顿第三: [%s]\n", energy, totalForce, status);

    LOG_PRINT("\n--- 20原子测试 ---\n");
    generatePositions(positions, 20, 15.0f, 42);
    ljForceCPU(positions, 20, epsilon, sigma, cutoff, forces, energy);
    float fx = 0, fy = 0, fz = 0;
    for (int i = 0; i < 20; i++) {
        fx += forces[i * 3];
        fy += forces[i * 3 + 1];
        fz += forces[i * 3 + 2];
    }
    status = (std::abs(fx) < 1e-4f && std::abs(fy) < 1e-4f && std::abs(fz) < 1e-4f) ? "PASS" : "FAIL";
    LOG_PRINT("  能量: %.6f, 总力: (%.6f, %.6f, %.6f), 牛顿第三: [%s]\n", energy, fx, fy, fz, status);
}

void testNPU(LJForceNPU& npu) {
    LOG_PRINT("\n============================================================\n");
    LOG_PRINT("NPU 融合算子测试\n");
    LOG_PRINT("============================================================\n");

    float sigma = 3.4f, epsilon = 0.01f, cutoff = 10.0f;

    int testCases[] = {20, 50, 100};
    const char* descs[] = {"小", "中", "大"};

    for (int t = 0; t < 3; t++) {
        int N = testCases[t];
        LOG_PRINT("\n--- %s规模: %d 原子 ---\n", descs[t], N);

        float boxSize = std::max(15.0f, std::pow((float)N, 1.0f/3.0f) * 4.0f);
        std::vector<float> positions;
        generatePositions(positions, N, boxSize, 42);

        // CPU 参考
        std::vector<float> forcesCPU;
        float energyCPU;
        ljForceCPU(positions, N, epsilon, sigma, cutoff, forcesCPU, energyCPU);

        // NPU
        std::vector<float> forcesNPU;
        float energyNPU;
        npu.compute(positions, N, epsilon, sigma, cutoff, forcesNPU, energyNPU);

        // 比较
        float maxForceErr = 0.0f, maxForceCPU = 0.0f;
        for (int i = 0; i < N * 3; i++) {
            maxForceErr = std::max(maxForceErr, std::abs(forcesNPU[i] - forcesCPU[i]));
            maxForceCPU = std::max(maxForceCPU, std::abs(forcesCPU[i]));
        }
        float forceRel = maxForceErr / (maxForceCPU + 1e-10f);
        float energyErr = std::abs(energyNPU - energyCPU);
        float energyRel = energyErr / (std::abs(energyCPU) + 1e-10f);

        const char* status = (forceRel < 0.01f && energyRel < 0.01f) ? "PASS" : "FAIL";
        LOG_PRINT("  力误差: %.2e (相对: %.2e)\n", maxForceErr, forceRel);
        LOG_PRINT("  能量: CPU=%.4f, NPU=%.4f, 误差=%.2e\n", energyCPU, energyNPU, energyRel);
        LOG_PRINT("  状态: [%s]\n", status);
    }
}

void benchmark(LJForceNPU& npu) {
    LOG_PRINT("\n============================================================\n");
    LOG_PRINT("性能对比 (CPU vs NPU)\n");
    LOG_PRINT("============================================================\n");

    float sigma = 3.4f, epsilon = 0.01f, cutoff = 10.0f;
    int testCases[] = {64, 128, 256};

    LOG_PRINT("\n%-8s %-10s %-12s %-12s %-10s\n", "N", "Pairs", "CPU(ms)", "NPU(ms)", "加速比");
    LOG_PRINT("----------------------------------------------------------------\n");

    for (int t = 0; t < 3; t++) {
        int N = testCases[t];
        int pairs = N * (N - 1) / 2;
        float boxSize = std::max(15.0f, std::pow((float)N, 1.0f/3.0f) * 4.0f);

        std::vector<float> positions;
        generatePositions(positions, N, boxSize, 42);

        std::vector<float> forces;
        float energy;

        // CPU 计时
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 3; i++) {
            ljForceCPU(positions, N, epsilon, sigma, cutoff, forces, energy);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double cpuMs = std::chrono::duration<double, std::milli>(end - start).count() / 3.0;

        // NPU 预热
        for (int i = 0; i < 3; i++) {
            npu.compute(positions, N, epsilon, sigma, cutoff, forces, energy);
        }

        // NPU 计时
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; i++) {
            npu.compute(positions, N, epsilon, sigma, cutoff, forces, energy);
        }
        end = std::chrono::high_resolution_clock::now();
        double npuMs = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;

        double speedup = cpuMs / npuMs;
        LOG_PRINT("%-8d %-10d %-12.2f %-12.2f %.2fx\n", N, pairs, cpuMs, npuMs, speedup);
    }
}

int main() {
    testCPU();

    LJForceNPU npu;
    if (!npu.init()) {
        LOG_PRINT("[ERROR] NPU 初始化失败\n");
        return 1;
    }

    testNPU(npu);
    benchmark(npu);

    LOG_PRINT("\n============================================================\n");
    LOG_PRINT("测试完成\n");
    LOG_PRINT("============================================================\n");

    return 0;
}
