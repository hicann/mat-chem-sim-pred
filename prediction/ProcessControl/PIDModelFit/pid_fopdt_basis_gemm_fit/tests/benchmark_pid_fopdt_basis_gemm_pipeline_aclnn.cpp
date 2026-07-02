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
#include "aclnn/acl_meta.h"
#include "aclnnop/level2/aclnn_matmul.h"

#ifndef PID_BASIS_PIPELINE_HOST_HEADER
#define PID_BASIS_PIPELINE_HOST_HEADER "pid_fopdt_basis_gemm_fit_host.h"
#endif
#include PID_BASIS_PIPELINE_HOST_HEADER

#ifndef PID_BASIS_PIPELINE_GET_WORKSPACE
#define PID_BASIS_PIPELINE_GET_WORKSPACE aclnnPidFopdtBasisGemmFitGetWorkspaceSize
#endif

#ifndef PID_BASIS_PIPELINE_RUN
#define PID_BASIS_PIPELINE_RUN aclnnPidFopdtBasisGemmFit
#endif

#ifndef PID_BASIS_PIPELINE_OP_NAME
#define PID_BASIS_PIPELINE_OP_NAME "PidFopdtBasisGemmPipeline"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct FitCase {
    std::vector<float> y_centered;
    std::vector<float> basis_t;
    std::vector<float> basis_norm;
    std::vector<float> y_energy;
    int64_t batch = 0;
    int64_t n = 0;
    int64_t candidates = 0;
    int64_t truth_index = 0;
};

struct FitOutput {
    std::vector<float> best_sse;
    std::vector<float> best_k;
    std::vector<int32_t> best_idx;
};

struct NpuTiming {
    FitOutput output;
    double resident_compute_ms = 0.0;
    double resident_d2h_ms = 0.0;
    double resident_e2e_ms = 0.0;
    double cold_h2d_ms = 0.0;
    double cold_compute_ms = 0.0;
    double cold_d2h_ms = 0.0;
    double cold_e2e_ms = 0.0;
    uint64_t matmul_workspace_bytes = 0;
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

void CheckAclnn(aclnnStatus ret, const char* expr)
{
    if (ret != OK) {
        throw std::runtime_error(std::string(expr) + " failed, ret=" + std::to_string(ret));
    }
}

#define CHECK_ACL(expr) CheckAcl((expr), #expr)
#define CHECK_ACLNN(expr) CheckAclnn((expr), #expr)

double MsSince(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void* MallocDevice(size_t bytes)
{
    if (bytes == 0U) {
        return nullptr;
    }
    void* ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

float SyntheticStepResponse(int64_t index, int64_t delay, float time_constant)
{
    if (index < delay) {
        return 0.0f;
    }
    const float t = static_cast<float>(index - delay + 1);
    return 1.0f - std::exp(-t / time_constant);
}

FitCase MakeCase(int64_t batch, int64_t n, int64_t candidates)
{
    FitCase data;
    data.batch = batch;
    data.n = n;
    data.candidates = candidates;
    data.truth_index = candidates / 2;
    data.y_centered.assign(static_cast<size_t>(batch * n), 0.0f);
    data.basis_t.assign(static_cast<size_t>(n * candidates), 0.0f);
    data.basis_norm.assign(static_cast<size_t>(candidates), 0.0f);
    data.y_energy.assign(static_cast<size_t>(batch), 0.0f);

    const int64_t max_delay = std::max<int64_t>(2, std::min<int64_t>(64, n / 8));
    constexpr float kTruthTimeConstant = 18.0f;
    constexpr int64_t kTruthDelay = 4;

    for (int64_t c = 0; c < candidates; ++c) {
        float time_constant = 6.0f + 0.17f * static_cast<float>((c * 37) % 251);
        int64_t delay = (c * 11) % max_delay;
        if (c == data.truth_index) {
            time_constant = kTruthTimeConstant;
            delay = kTruthDelay;
        } else if (std::fabs(time_constant - kTruthTimeConstant) < 1.0e-3f && delay == kTruthDelay) {
            delay = (delay + 1) % max_delay;
        }

        float norm = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float value = SyntheticStepResponse(i, delay, time_constant);
            if (c != data.truth_index) {
                const float ripple =
                    0.003f * std::sin(0.011f * static_cast<float>((i + 1) * ((c % 17) + 1)));
                value += ripple;
            }
            data.basis_t[static_cast<size_t>(i * candidates + c)] = value;
            norm += value * value;
        }
        data.basis_norm[static_cast<size_t>(c)] = std::max(norm, 1.0e-6f);
    }

    for (int64_t b = 0; b < batch; ++b) {
        const float gain = 1.2f + 0.03f * static_cast<float>(b % 11);
        float energy = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            const float base = data.basis_t[static_cast<size_t>(i * candidates + data.truth_index)];
            const float noise = 0.002f * std::sin(0.017f * static_cast<float>((i + 1) * ((b % 7) + 1)));
            const float value = gain * base + noise;
            data.y_centered[static_cast<size_t>(b * n + i)] = value;
            energy += value * value;
        }
        data.y_energy[static_cast<size_t>(b)] = energy;
    }
    return data;
}

void CpuFitRange(const FitCase& data, int64_t begin, int64_t end, FitOutput& out)
{
    constexpr float kEpsilon = 1.0e-6f;
    constexpr float kLargeSse = std::numeric_limits<float>::max();
    for (int64_t b = begin; b < end; ++b) {
        const float* y = data.y_centered.data() + static_cast<size_t>(b * data.n);
        const float energy = data.y_energy[static_cast<size_t>(b)];
        float best_sse = kLargeSse;
        float best_k = 0.0f;
        int32_t best_idx = 0;
        for (int64_t c = 0; c < data.candidates; ++c) {
            float dot = 0.0f;
            for (int64_t i = 0; i < data.n; ++i) {
                dot += y[static_cast<size_t>(i)] *
                       data.basis_t[static_cast<size_t>(i * data.candidates + c)];
            }

            const float norm = data.basis_norm[static_cast<size_t>(c)];
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

FitOutput CpuFit(const FitCase& data, int threads)
{
    FitOutput out;
    out.best_sse.assign(static_cast<size_t>(data.batch), 0.0f);
    out.best_k.assign(static_cast<size_t>(data.batch), 0.0f);
    out.best_idx.assign(static_cast<size_t>(data.batch), 0);
    const int worker_count = std::max(1, threads);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int worker = 0; worker < worker_count; ++worker) {
        const int64_t begin = data.batch * worker / worker_count;
        const int64_t end = data.batch * (worker + 1) / worker_count;
        workers.emplace_back([&data, begin, end, &out]() { CpuFitRange(data, begin, end, out); });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return out;
}

double MeasureCpuMs(const FitCase& data, int threads, int iters)
{
    FitOutput guard;
    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        guard = CpuFit(data, threads);
    }
    auto end = Clock::now();
    if (!guard.best_sse.empty()) {
        std::cout << "cpu_guard=" << guard.best_sse[0] << std::endl;
    }
    return MsSince(start, end) / static_cast<double>(iters);
}

aclTensor* CreateTensor2D(void* data, int64_t rows, int64_t cols)
{
    const int64_t dims[2] = {rows, cols};
    const int64_t strides[2] = {cols, 1};
    return aclCreateTensor(dims, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, dims, 2, data);
}

NpuTiming RunNpu(const FitCase& data, int device_id, int iters, int warmup)
{
    NpuTiming timing;
    timing.output.best_sse.assign(static_cast<size_t>(data.batch), 0.0f);
    timing.output.best_k.assign(static_cast<size_t>(data.batch), 0.0f);
    timing.output.best_idx.assign(static_cast<size_t>(data.batch), 0);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* d_y = MallocDevice(data.y_centered.size() * sizeof(float));
    void* d_basis = MallocDevice(data.basis_t.size() * sizeof(float));
    void* d_dot = MallocDevice(static_cast<size_t>(data.batch * data.candidates) * sizeof(float));
    void* d_norm = MallocDevice(data.basis_norm.size() * sizeof(float));
    void* d_energy = MallocDevice(data.y_energy.size() * sizeof(float));
    void* d_sse = MallocDevice(timing.output.best_sse.size() * sizeof(float));
    void* d_k = MallocDevice(timing.output.best_k.size() * sizeof(float));
    void* d_idx = MallocDevice(timing.output.best_idx.size() * sizeof(int32_t));
    const uint64_t reduce_workspace_size =
        PID_BASIS_PIPELINE_GET_WORKSPACE(data.batch, data.candidates);
    void* reduce_workspace = MallocDevice(reduce_workspace_size);

    aclTensor* y_tensor = CreateTensor2D(d_y, data.batch, data.n);
    aclTensor* basis_tensor = CreateTensor2D(d_basis, data.n, data.candidates);
    aclTensor* dot_tensor = CreateTensor2D(d_dot, data.batch, data.candidates);
    if (y_tensor == nullptr || basis_tensor == nullptr || dot_tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor failed");
    }

    uint64_t matmul_workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACLNN(aclnnMatmulGetWorkspaceSize(
        y_tensor, basis_tensor, dot_tensor, 0, &matmul_workspace_size, &executor));
    timing.matmul_workspace_bytes = matmul_workspace_size;
    void* matmul_workspace = MallocDevice(matmul_workspace_size);

    auto copy_inputs = [&]() {
        CHECK_ACL(aclrtMemcpy(d_y, data.y_centered.size() * sizeof(float), data.y_centered.data(),
                              data.y_centered.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_basis, data.basis_t.size() * sizeof(float), data.basis_t.data(),
                              data.basis_t.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_norm, data.basis_norm.size() * sizeof(float), data.basis_norm.data(),
                              data.basis_norm.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_energy, data.y_energy.size() * sizeof(float), data.y_energy.data(),
                              data.y_energy.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    };
    auto launch_pipeline = [&]() {
        uint64_t workspace_size = 0;
        aclOpExecutor* current_executor = nullptr;
        CHECK_ACLNN(aclnnMatmulGetWorkspaceSize(
            y_tensor, basis_tensor, dot_tensor, 0, &workspace_size, &current_executor));
        if (workspace_size > matmul_workspace_size) {
            throw std::runtime_error("aclnnMatmul workspace size grew unexpectedly");
        }
        CHECK_ACLNN(aclnnMatmul(matmul_workspace, workspace_size, current_executor, stream));
        const int32_t ret = PID_BASIS_PIPELINE_RUN(
            d_dot, d_norm, d_energy, d_sse, d_k, d_idx, data.batch, data.candidates, reduce_workspace,
            reduce_workspace_size, stream);
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
        launch_pipeline();
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        launch_pipeline();
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }
    auto end = Clock::now();
    timing.resident_compute_ms = MsSince(start, end) / static_cast<double>(iters);

    double resident_d2h_total = 0.0;
    double resident_e2e_total = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto e2e_start = Clock::now();
        launch_pipeline();
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto section_start = Clock::now();
        copy_outputs();
        auto section_end = Clock::now();
        resident_d2h_total += MsSince(section_start, section_end);
        resident_e2e_total += MsSince(e2e_start, section_end);
    }
    timing.resident_d2h_ms = resident_d2h_total / static_cast<double>(iters);
    timing.resident_e2e_ms = resident_e2e_total / static_cast<double>(iters);

    double cold_h2d_total = 0.0;
    double cold_compute_total = 0.0;
    double cold_d2h_total = 0.0;
    double cold_e2e_total = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto e2e_start = Clock::now();
        auto section_start = Clock::now();
        copy_inputs();
        auto section_end = Clock::now();
        cold_h2d_total += MsSince(section_start, section_end);

        section_start = Clock::now();
        launch_pipeline();
        CHECK_ACL(aclrtSynchronizeStream(stream));
        section_end = Clock::now();
        cold_compute_total += MsSince(section_start, section_end);

        section_start = Clock::now();
        copy_outputs();
        section_end = Clock::now();
        cold_d2h_total += MsSince(section_start, section_end);
        cold_e2e_total += MsSince(e2e_start, section_end);
    }
    timing.cold_h2d_ms = cold_h2d_total / static_cast<double>(iters);
    timing.cold_compute_ms = cold_compute_total / static_cast<double>(iters);
    timing.cold_d2h_ms = cold_d2h_total / static_cast<double>(iters);
    timing.cold_e2e_ms = cold_e2e_total / static_cast<double>(iters);
    copy_outputs();

    aclDestroyTensor(dot_tensor);
    aclDestroyTensor(basis_tensor);
    aclDestroyTensor(y_tensor);
    aclrtFree(matmul_workspace);
    aclrtFree(reduce_workspace);
    aclrtFree(d_idx);
    aclrtFree(d_k);
    aclrtFree(d_sse);
    aclrtFree(d_energy);
    aclrtFree(d_norm);
    aclrtFree(d_dot);
    aclrtFree(d_basis);
    aclrtFree(d_y);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return timing;
}

ErrorStats ComputeErrors(const FitOutput& cpu, const FitOutput& npu)
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
                  << " <device_id> [batch=64] [n=1024] [candidates=256] [iters=5] [warmup=2]"
                     " [cpu_threads=hardware]"
                  << std::endl;
        return 1;
    }
    const int device_id = std::atoi(argv[1]);
    const int64_t batch = argc > 2 ? std::atoll(argv[2]) : 64;
    const int64_t n = argc > 3 ? std::atoll(argv[3]) : 1024;
    const int64_t candidates = argc > 4 ? std::atoll(argv[4]) : 256;
    const int iters = argc > 5 ? std::atoi(argv[5]) : 5;
    const int warmup = argc > 6 ? std::atoi(argv[6]) : 2;
    const int cpu_threads = ResolveThreads(argc > 7 ? std::atoi(argv[7]) : 0);
    if (batch <= 0 || n <= 0 || candidates <= 0 || iters <= 0 || warmup < 0) {
        std::cerr << "batch/n/candidates/iters must be positive; warmup must be non-negative" << std::endl;
        return 1;
    }

    try {
        const FitCase data = MakeCase(batch, n, candidates);
        const FitOutput cpu = CpuFit(data, cpu_threads);
        const double cpu_single_ms = MeasureCpuMs(data, 1, std::max(1, std::min(3, iters)));
        const double cpu_parallel_ms = MeasureCpuMs(data, cpu_threads, iters);
        const NpuTiming npu = RunNpu(data, device_id, iters, warmup);
        const ErrorStats err = ComputeErrors(cpu, npu.output);

        std::cout << "op=" << PID_BASIS_PIPELINE_OP_NAME << " batch=" << batch << " n=" << n
                  << " candidates=" << candidates << " iters=" << iters << " warmup=" << warmup << std::endl;
        std::cout << "cpu_threads=" << cpu_threads << std::endl;
        std::cout << "matmul_flops=" << (2.0 * static_cast<double>(batch) * static_cast<double>(n) *
                                          static_cast<double>(candidates))
                  << std::endl;
        std::cout << "cpu_single_ms_avg=" << cpu_single_ms << std::endl;
        std::cout << "cpu_parallel_ms_avg=" << cpu_parallel_ms << std::endl;
        std::cout << "npu_matmul_workspace_bytes=" << npu.matmul_workspace_bytes << std::endl;
        std::cout << "npu_resident_compute_ms_avg=" << npu.resident_compute_ms << std::endl;
        std::cout << "npu_resident_d2h_ms_avg=" << npu.resident_d2h_ms << std::endl;
        std::cout << "npu_resident_e2e_ms_avg=" << npu.resident_e2e_ms << std::endl;
        std::cout << "npu_cold_h2d_ms_avg=" << npu.cold_h2d_ms << std::endl;
        std::cout << "npu_cold_compute_ms_avg=" << npu.cold_compute_ms << std::endl;
        std::cout << "npu_cold_d2h_ms_avg=" << npu.cold_d2h_ms << std::endl;
        std::cout << "npu_cold_e2e_ms_avg=" << npu.cold_e2e_ms << std::endl;
        std::cout << "speedup_npu_resident_compute_vs_cpu_parallel="
                  << (cpu_parallel_ms / npu.resident_compute_ms) << std::endl;
        std::cout << "speedup_npu_resident_e2e_vs_cpu_parallel=" << (cpu_parallel_ms / npu.resident_e2e_ms)
                  << std::endl;
        std::cout << "speedup_npu_cold_e2e_vs_cpu_parallel=" << (cpu_parallel_ms / npu.cold_e2e_ms) << std::endl;
        std::cout << "max_abs_sse=" << err.max_abs_sse << std::endl;
        std::cout << "max_rel_sse=" << err.max_rel_sse << std::endl;
        std::cout << "max_abs_k=" << err.max_abs_k << std::endl;
        std::cout << "idx_diff_count=" << err.idx_diff << std::endl;
        return (err.max_rel_sse < 5.0e-2f && err.max_abs_k < 1.0e-2f && err.idx_diff == 0) ? 0 : 2;
    } catch (const std::exception& ex) {
        std::cerr << "benchmark failed: " << ex.what() << std::endl;
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }
}
