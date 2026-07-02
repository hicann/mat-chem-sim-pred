/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "pid_ipdt_basis_gemm_fit_host.h"

namespace {

using Clock = std::chrono::steady_clock;

struct ReduceCase {
    std::vector<float> dot;
    std::vector<float> basis_norm;
    std::vector<float> y_energy;
    int64_t batch = 0;
    int64_t candidates = 0;
};

struct ReduceOutput {
    std::vector<float> best_sse;
    std::vector<float> best_k;
    std::vector<int32_t> best_idx;
};

struct NpuTiming {
    ReduceOutput output;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double e2e_ms = 0.0;
};

struct ErrorStats {
    float max_abs_sse = 0.0f;
    float max_rel_sse = 0.0f;
    float max_abs_k = 0.0f;
    int idx_diff = 0;
};

void CheckAcl(aclError ret, const char* expr)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(expr) + " failed, ret=" + std::to_string(ret));
    }
}

#define CHECK_ACL(expr) CheckAcl((expr), #expr)

double MsSince(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void* MallocDevice(size_t bytes)
{
    void* ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

ReduceCase MakeCase(int64_t batch, int64_t candidates)
{
    ReduceCase data;
    data.batch = batch;
    data.candidates = candidates;
    data.dot.assign(static_cast<size_t>(batch * candidates), 0.0f);
    data.basis_norm.assign(static_cast<size_t>(candidates), 0.0f);
    data.y_energy.assign(static_cast<size_t>(batch), 0.0f);

    for (int64_t c = 0; c < candidates; ++c) {
        data.basis_norm[static_cast<size_t>(c)] = 1.0f + 0.01f * static_cast<float>(c % 251);
    }

    for (int64_t b = 0; b < batch; ++b) {
        const int64_t best = (b * 131 + 17) % candidates;
        const float energy = 1000.0f + 0.5f * static_cast<float>(b % 127);
        data.y_energy[static_cast<size_t>(b)] = energy;
        for (int64_t c = 0; c < candidates; ++c) {
            const float norm = data.basis_norm[static_cast<size_t>(c)];
            const float small = static_cast<float>(((b + 3) * (c + 11)) % 97) * 0.01f;
            data.dot[static_cast<size_t>(b * candidates + c)] = small * norm;
        }
        const float best_norm = data.basis_norm[static_cast<size_t>(best)];
        data.dot[static_cast<size_t>(b * candidates + best)] = std::sqrt(energy * best_norm) * 0.95f;
    }
    return data;
}

void CpuReduceRange(const ReduceCase& data, int64_t begin, int64_t end, ReduceOutput& out)
{
    constexpr float kEpsilon = 1.0e-6f;
    constexpr float kLargeSse = std::numeric_limits<float>::max();
    for (int64_t b = begin; b < end; ++b) {
        const float energy = data.y_energy[static_cast<size_t>(b)];
        float best_sse = kLargeSse;
        float best_k = 0.0f;
        int32_t best_idx = 0;
        for (int64_t c = 0; c < data.candidates; ++c) {
            const float norm = data.basis_norm[static_cast<size_t>(c)];
            const float dot = data.dot[static_cast<size_t>(b * data.candidates + c)];
            float gain = 0.0f;
            float sse = energy;
            if (norm > kEpsilon) {
                gain = dot / norm;
                sse = energy - dot * dot / norm;
                if (sse < 0.0f && sse > -1.0e-3f) {
                    sse = 0.0f;
                }
            }
            if (!std::isfinite(sse)) {
                sse = kLargeSse;
                gain = 0.0f;
            }
            if (sse < best_sse) {
                best_sse = sse;
                best_k = gain;
                best_idx = static_cast<int32_t>(c);
            }
        }
        out.best_sse[static_cast<size_t>(b)] = best_sse;
        out.best_k[static_cast<size_t>(b)] = best_k;
        out.best_idx[static_cast<size_t>(b)] = best_idx;
    }
}

ReduceOutput CpuReduce(const ReduceCase& data, int threads)
{
    ReduceOutput out;
    out.best_sse.assign(static_cast<size_t>(data.batch), 0.0f);
    out.best_k.assign(static_cast<size_t>(data.batch), 0.0f);
    out.best_idx.assign(static_cast<size_t>(data.batch), 0);
    const int worker_count = std::max(1, threads);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int worker = 0; worker < worker_count; ++worker) {
        const int64_t begin = data.batch * worker / worker_count;
        const int64_t end = data.batch * (worker + 1) / worker_count;
        workers.emplace_back([&data, begin, end, &out]() { CpuReduceRange(data, begin, end, out); });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return out;
}

double MeasureCpuMs(const ReduceCase& data, int threads, int iters)
{
    ReduceOutput guard;
    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        guard = CpuReduce(data, threads);
    }
    auto end = Clock::now();
    if (!guard.best_sse.empty()) {
        std::cout << "cpu_guard=" << guard.best_sse[0] << std::endl;
    }
    return MsSince(start, end) / static_cast<double>(iters);
}

NpuTiming RunNpu(const ReduceCase& data, int device_id, int iters, int warmup)
{
    NpuTiming timing;
    timing.output.best_sse.assign(static_cast<size_t>(data.batch), 0.0f);
    timing.output.best_k.assign(static_cast<size_t>(data.batch), 0.0f);
    timing.output.best_idx.assign(static_cast<size_t>(data.batch), 0);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* d_dot = MallocDevice(data.dot.size() * sizeof(float));
    void* d_norm = MallocDevice(data.basis_norm.size() * sizeof(float));
    void* d_energy = MallocDevice(data.y_energy.size() * sizeof(float));
    void* d_sse = MallocDevice(timing.output.best_sse.size() * sizeof(float));
    void* d_k = MallocDevice(timing.output.best_k.size() * sizeof(float));
    void* d_idx = MallocDevice(timing.output.best_idx.size() * sizeof(int32_t));
    const uint64_t workspace_size = aclnnPidIpdtBasisGemmFitGetWorkspaceSize(data.batch, data.candidates);
    void* workspace = MallocDevice(workspace_size);

    auto copy_inputs = [&]() {
        CHECK_ACL(aclrtMemcpy(d_dot, data.dot.size() * sizeof(float), data.dot.data(),
                              data.dot.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_norm, data.basis_norm.size() * sizeof(float), data.basis_norm.data(),
                              data.basis_norm.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_energy, data.y_energy.size() * sizeof(float), data.y_energy.data(),
                              data.y_energy.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    };
    auto launch_once = [&]() {
        const int32_t ret = aclnnPidIpdtBasisGemmFit(
            d_dot, d_norm, d_energy, d_sse, d_k, d_idx, data.batch, data.candidates, workspace, workspace_size, stream);
        CHECK_ACL(static_cast<aclError>(ret));
    };
    auto copy_outputs = [&]() {
        CHECK_ACL(aclrtMemcpy(timing.output.best_sse.data(), timing.output.best_sse.size() * sizeof(float), d_sse,
                              timing.output.best_sse.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(timing.output.best_k.data(), timing.output.best_k.size() * sizeof(float), d_k,
                              timing.output.best_k.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(timing.output.best_idx.data(), timing.output.best_idx.size() * sizeof(int32_t), d_idx,
                              timing.output.best_idx.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    };

    copy_inputs();
    for (int i = 0; i < warmup; ++i) {
        launch_once();
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        launch_once();
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }
    auto end = Clock::now();
    timing.kernel_ms = MsSince(start, end) / static_cast<double>(iters);

    double h2d_total = 0.0;
    double kernel_total = 0.0;
    double d2h_total = 0.0;
    double e2e_total = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto e2e_start = Clock::now();
        auto section_start = Clock::now();
        copy_inputs();
        auto section_end = Clock::now();
        h2d_total += MsSince(section_start, section_end);

        section_start = Clock::now();
        launch_once();
        CHECK_ACL(aclrtSynchronizeStream(stream));
        section_end = Clock::now();
        kernel_total += MsSince(section_start, section_end);

        section_start = Clock::now();
        copy_outputs();
        section_end = Clock::now();
        d2h_total += MsSince(section_start, section_end);
        e2e_total += MsSince(e2e_start, section_end);
    }
    timing.h2d_ms = h2d_total / static_cast<double>(iters);
    timing.kernel_ms = kernel_total / static_cast<double>(iters);
    timing.d2h_ms = d2h_total / static_cast<double>(iters);
    timing.e2e_ms = e2e_total / static_cast<double>(iters);
    copy_outputs();

    aclrtFree(workspace);
    aclrtFree(d_idx);
    aclrtFree(d_k);
    aclrtFree(d_sse);
    aclrtFree(d_energy);
    aclrtFree(d_norm);
    aclrtFree(d_dot);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return timing;
}

ErrorStats ComputeErrors(const ReduceOutput& cpu, const ReduceOutput& npu)
{
    ErrorStats stats;
    for (size_t i = 0; i < cpu.best_sse.size(); ++i) {
        const float abs_sse = std::fabs(cpu.best_sse[i] - npu.best_sse[i]);
        const float rel_sse = abs_sse / std::max(std::fabs(cpu.best_sse[i]), 1.0f);
        stats.max_abs_sse = std::max(stats.max_abs_sse, abs_sse);
        stats.max_rel_sse = std::max(stats.max_rel_sse, rel_sse);
        stats.max_abs_k = std::max(stats.max_abs_k, std::fabs(cpu.best_k[i] - npu.best_k[i]));
        if (cpu.best_idx[i] != npu.best_idx[i]) {
            ++stats.idx_diff;
        }
    }
    return stats;
}

int ResolveThreads(int requested)
{
    if (requested > 0) {
        return requested;
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <device_id> [batch=1024] [candidates=4096] [iters=10] [warmup=2] [cpu_threads=hardware]"
                  << std::endl;
        return 1;
    }
    const int device_id = std::atoi(argv[1]);
    const int64_t batch = argc > 2 ? std::atoll(argv[2]) : 1024;
    const int64_t candidates = argc > 3 ? std::atoll(argv[3]) : 4096;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 10;
    const int warmup = argc > 5 ? std::atoi(argv[5]) : 2;
    const int cpu_threads = ResolveThreads(argc > 6 ? std::atoi(argv[6]) : 0);
    if (batch <= 0 || candidates <= 0 || iters <= 0 || warmup < 0) {
        std::cerr << "batch/candidates/iters must be positive; warmup must be non-negative" << std::endl;
        return 1;
    }

    try {
        const ReduceCase data = MakeCase(batch, candidates);
        const ReduceOutput cpu = CpuReduce(data, cpu_threads);
        const double cpu_single_ms = MeasureCpuMs(data, 1, std::max(1, std::min(3, iters)));
        const double cpu_parallel_ms = MeasureCpuMs(data, cpu_threads, iters);
        const NpuTiming npu = RunNpu(data, device_id, iters, warmup);
        const ErrorStats err = ComputeErrors(cpu, npu.output);

        std::cout << "op=PidIpdtBasisGemmFitReduce batch=" << batch << " candidates=" << candidates
                  << " iters=" << iters << " warmup=" << warmup << std::endl;
        std::cout << "cpu_threads=" << cpu_threads << std::endl;
        std::cout << "work_items=" << static_cast<double>(batch) * static_cast<double>(candidates) << std::endl;
        std::cout << "cpu_single_ms_avg=" << cpu_single_ms << std::endl;
        std::cout << "cpu_parallel_ms_avg=" << cpu_parallel_ms << std::endl;
        std::cout << "npu_h2d_ms_avg=" << npu.h2d_ms << std::endl;
        std::cout << "npu_kernel_ms_avg=" << npu.kernel_ms << std::endl;
        std::cout << "npu_d2h_ms_avg=" << npu.d2h_ms << std::endl;
        std::cout << "npu_e2e_ms_avg=" << npu.e2e_ms << std::endl;
        std::cout << "speedup_npu_kernel_vs_cpu_parallel=" << (cpu_parallel_ms / npu.kernel_ms) << std::endl;
        std::cout << "speedup_npu_e2e_vs_cpu_parallel=" << (cpu_parallel_ms / npu.e2e_ms) << std::endl;
        std::cout << "max_abs_sse=" << err.max_abs_sse << std::endl;
        std::cout << "max_rel_sse=" << err.max_rel_sse << std::endl;
        std::cout << "max_abs_k=" << err.max_abs_k << std::endl;
        std::cout << "idx_diff_count=" << err.idx_diff << std::endl;
        return (err.max_rel_sse < 1.0e-3f && err.max_abs_k < 1.0e-3f && err.idx_diff == 0) ? 0 : 2;
    } catch (const std::exception& ex) {
        std::cerr << "benchmark failed: " << ex.what() << std::endl;
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }
}
