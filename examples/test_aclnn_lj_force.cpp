/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// 外部声明
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

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int main()
{
    // 1. 初始化设备
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入数据
    const int32_t numAtoms = 64;
    const float epsilon = 0.0103f;  // eV (Argon)
    const float sigma = 3.4f;       // Angstrom
    const float cutoff = 10.0f;     // Angstrom

    // 生成随机原子坐标 (简单立方晶格)
    std::vector<float> positions(numAtoms * 3);
    float spacing = 4.0f;
    int idx = 0;
    for (int i = 0; i < 4 && idx < numAtoms; i++) {
        for (int j = 0; j < 4 && idx < numAtoms; j++) {
            for (int k = 0; k < 4 && idx < numAtoms; k++) {
                positions[idx * 3] = i * spacing;
                positions[idx * 3 + 1] = j * spacing;
                positions[idx * 3 + 2] = k * spacing;
                idx++;
            }
        }
    }

    std::vector<float> forces(numAtoms * 3, 0.0f);
    std::vector<float> energy(32, 0.0f);  // 每核一个能量槽

    // 3. 分配设备内存
    void* posDeviceAddr = nullptr;
    void* forcesDeviceAddr = nullptr;
    void* energyDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;

    size_t posSize = numAtoms * 3 * sizeof(float);
    size_t forcesSize = numAtoms * 3 * sizeof(float) + 32 * 8 * sizeof(float);  // 含对齐余量
    size_t energySize = 32 * 8 * sizeof(float);
    uint64_t workspaceSize = aclnnLJForceGetWorkspaceSize(numAtoms);

    ret = aclrtMalloc(&posDeviceAddr, posSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc positions failed\n"); return ret);

    ret = aclrtMalloc(&forcesDeviceAddr, forcesSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc forces failed\n"); return ret);

    ret = aclrtMalloc(&energyDeviceAddr, energySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc energy failed\n"); return ret);

    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc workspace failed\n"); return ret);

    // 4. 拷贝输入到设备
    ret = aclrtMemcpy(posDeviceAddr, posSize, positions.data(), posSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("memcpy positions failed\n"); return ret);

    // 5. 调用算子
    ret = aclnnLJForceDirect(
        posDeviceAddr,
        forcesDeviceAddr,
        energyDeviceAddr,
        numAtoms,
        epsilon,
        sigma,
        cutoff,
        workspaceAddr,
        workspaceSize,
        stream
    );
    CHECK_RET(ret == 0, LOG_PRINT("aclnnLJForceDirect failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed\n"); return ret);

    // 7. 拷贝结果回主机
    ret = aclrtMemcpy(forces.data(), forces.size() * sizeof(float),
                      forcesDeviceAddr, numAtoms * 3 * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("memcpy forces failed\n"); return ret);

    ret = aclrtMemcpy(energy.data(), energy.size() * sizeof(float),
                      energyDeviceAddr, energySize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("memcpy energy failed\n"); return ret);

    // 8. 输出结果
    float totalEnergy = 0.0f;
    for (int i = 0; i < 32; i++) {
        totalEnergy += energy[i * 8];
    }
    LOG_PRINT("Total potential energy: %f eV\n", totalEnergy);
    LOG_PRINT("First atom force: (%f, %f, %f) eV/A\n",
              forces[0], forces[1], forces[2]);

    // 9. 释放资源
    aclrtFree(posDeviceAddr);
    aclrtFree(forcesDeviceAddr);
    aclrtFree(energyDeviceAddr);
    aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("LJForceFused test completed successfully.\n");
    return 0;
}
