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
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "pid_step_response_features_host.h"

namespace {

using Clock = std::chrono::steady_clock;

struct CaseData {
    std::vector<float> pv;
    std::vector<float> sp;
    int64_t batch = 0;
    int64_t candidates = 0;
    int64_t sample_count = 0;
    float sample_interval = 1.0f;
    float settle_band_ratio = 0.02f;
};

struct NpuResult {
    std::vector<float> features;
    double kernel_ms = 0.0;
    double resident_e2e_ms = 0.0;
};

struct ErrorStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
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

float AbsF(float value)
{
    return value >= 0.0f ? value : -value;
}

void* MallocDevice(size_t bytes)
{
    void* ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

CaseData MakeCase(int64_t batch, int64_t candidates, int64_t sample_count)
{
    CaseData data;
    data.batch = batch;
    data.candidates = candidates;
    data.sample_count = sample_count;
    data.sample_interval = 1.0f;
    data.settle_band_ratio = 0.02f;
    data.pv.assign(static_cast<size_t>(batch * candidates * sample_count), 0.0f);
    data.sp.assign(static_cast<size_t>(batch * sample_count), 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        const float sp_initial = 45.0f + 0.05f * static_cast<float>(b % 7);
        const float sp_final = 50.0f + 0.1f * static_cast<float>(b % 5);
        for (int64_t i = 0; i < sample_count; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(std::max<int64_t>(1, sample_count - 1));
            data.sp[static_cast<size_t>(b * sample_count + i)] = t >= 0.05f ? sp_final : sp_initial;
        }
        const float target_delta = sp_final - sp_initial;
        for (int64_t c = 0; c < candidates; ++c) {
            const float cand = static_cast<float>(c);
            const float tau = 0.08f + 0.22f * (cand + 1.0f) / static_cast<float>(std::max<int64_t>(1, candidates));
            const float damping = 0.04f + 0.18f * static_cast<float>(c % 9) / 8.0f;
            for (int64_t i = 0; i < sample_count; ++i) {
                const float t =
                    static_cast<float>(i) / static_cast<float>(std::max<int64_t>(1, sample_count - 1));
                const float shifted = std::max(t - 0.05f, 0.0f);
                const float response = 1.0f - std::exp(-shifted / tau);
                const float ring = damping * std::exp(-3.0f * shifted) *
                                   std::sin(6.283185307179586f * (3.0f + cand * 0.03f) * t);
                const size_t offset = static_cast<size_t>((b * candidates + c) * sample_count + i);
                data.pv[offset] = sp_initial + target_delta * (response + ring);
            }
        }
    }
    return data;
}

void ComputeTask(const CaseData& data, int64_t task, std::vector<float>& out)
{
    const int64_t b = task / data.candidates;
    const int64_t c = task - b * data.candidates;
    const size_t pv_base = static_cast<size_t>((b * data.candidates + c) * data.sample_count);
    const size_t sp_base = static_cast<size_t>(b * data.sample_count);
    const float target = data.sp[sp_base + static_cast<size_t>(data.sample_count - 1)];
    const float initial = data.pv[pv_base];
    const float final_value = data.pv[pv_base + static_cast<size_t>(data.sample_count - 1)];
    const float delta = target - initial;
    const float abs_delta = std::max(AbsF(delta), 1.0e-6f);
    const float direction = delta >= 0.0f ? 1.0f : -1.0f;
    const float band = std::max(abs_delta * data.settle_band_ratio, 1.0e-4f);

    float peak = initial;
    float trough = initial;
    int64_t peak_idx = 0;
    int64_t rise10 = data.sample_count - 1;
    int64_t rise90 = data.sample_count - 1;
    bool found10 = false;
    bool found90 = false;
    int64_t last_unsettled = -1;
    float iae = 0.0f;
    float ise = 0.0f;
    for (int64_t i = 0; i < data.sample_count; ++i) {
        const float pv_value = data.pv[pv_base + static_cast<size_t>(i)];
        if (pv_value > peak) {
            peak = pv_value;
            peak_idx = i;
        }
        trough = std::min(trough, pv_value);
        const float normalized = direction * (pv_value - initial) / abs_delta;
        if (!found10 && normalized >= 0.1f) {
            rise10 = i;
            found10 = true;
        }
        if (!found90 && normalized >= 0.9f) {
            rise90 = i;
            found90 = true;
        }
        const float error = AbsF(data.sp[sp_base + static_cast<size_t>(i)] - pv_value);
        iae += error * data.sample_interval;
        ise += error * error * data.sample_interval;
        if (error > band) {
            last_unsettled = i;
        }
    }

    const size_t out_base = static_cast<size_t>(task * PID_STEP_RESPONSE_FEATURE_COUNT);
    out[out_base + 0U] = initial;
    out[out_base + 1U] = final_value;
    out[out_base + 2U] = AbsF(target - final_value);
    out[out_base + 3U] = peak;
    out[out_base + 4U] = trough;
    out[out_base + 5U] = std::max(direction * (peak - target), 0.0f) / abs_delta;
    out[out_base + 6U] = std::max(direction * (target - trough), 0.0f) / abs_delta;
    out[out_base + 7U] = std::max<int64_t>(rise90 - rise10, 0) * data.sample_interval;
    out[out_base + 8U] = static_cast<float>(peak_idx) * data.sample_interval;
    out[out_base + 9U] = last_unsettled >= 0 ? static_cast<float>(last_unsettled + 1) * data.sample_interval : 0.0f;
    out[out_base + 10U] = iae;
    out[out_base + 11U] = ise;
}

std::vector<float> CpuReference(const CaseData& data, int threads)
{
    const int64_t total_tasks = data.batch * data.candidates;
    const int worker_count = static_cast<int>(std::min<int64_t>(std::max(1, threads), total_tasks));
    std::vector<float> out(static_cast<size_t>(total_tasks * PID_STEP_RESPONSE_FEATURE_COUNT), 0.0f);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int worker = 0; worker < worker_count; ++worker) {
        const int64_t begin = total_tasks * worker / worker_count;
        const int64_t end = total_tasks * (worker + 1) / worker_count;
        workers.emplace_back([&, begin, end]() {
            for (int64_t task = begin; task < end; ++task) {
                ComputeTask(data, task, out);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return out;
}

double MeasureCpuMs(const CaseData& data, int threads, int iters)
{
    volatile double guard = 0.0;
    const auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        auto out = CpuReference(data, threads);
        guard += out[static_cast<size_t>(i) % out.size()];
    }
    const auto end = Clock::now();
    std::cout << "cpu_guard=" << guard << std::endl;
    return MsSince(start, end) / static_cast<double>(iters);
}

ErrorStats Compare(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    ErrorStats stats;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float abs_err = AbsF(lhs[i] - rhs[i]);
        const float rel_err = abs_err / std::max(AbsF(rhs[i]), 1.0e-6f);
        stats.max_abs = std::max(stats.max_abs, abs_err);
        stats.max_rel = std::max(stats.max_rel, rel_err);
    }
    return stats;
}

NpuResult RunNpu(const CaseData& data, int device_id, int iters)
{
    const size_t pv_bytes = data.pv.size() * sizeof(float);
    const size_t sp_bytes = data.sp.size() * sizeof(float);
    const size_t features_bytes =
        static_cast<size_t>(data.batch * data.candidates * PID_STEP_RESPONSE_FEATURE_COUNT) * sizeof(float);

    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    void* d_pv = MallocDevice(pv_bytes);
    void* d_sp = MallocDevice(sp_bytes);
    void* d_features = MallocDevice(features_bytes);
    const uint64_t workspace_size =
        aclnnPidStepResponseFeaturesGetWorkspaceSize(data.batch, data.candidates, data.sample_count);
    void* workspace = MallocDevice(workspace_size);

    CHECK_ACL(aclrtMemcpy(d_pv, pv_bytes, data.pv.data(), pv_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_sp, sp_bytes, data.sp.data(), sp_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(static_cast<aclError>(aclnnPidStepResponseFeatures(
        d_pv, d_sp, d_features, data.batch, data.candidates, data.sample_count, data.sample_interval,
        data.settle_band_ratio, workspace, workspace_size, stream)));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    double kernel_total = 0.0;
    for (int i = 0; i < iters; ++i) {
        const auto start = Clock::now();
        CHECK_ACL(static_cast<aclError>(aclnnPidStepResponseFeatures(
            d_pv, d_sp, d_features, data.batch, data.candidates, data.sample_count, data.sample_interval,
            data.settle_band_ratio, workspace, workspace_size, stream)));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        kernel_total += MsSince(start, Clock::now());
    }

    NpuResult result;
    result.features.assign(features_bytes / sizeof(float), 0.0f);
    double e2e_total = 0.0;
    for (int i = 0; i < iters; ++i) {
        const auto start = Clock::now();
        CHECK_ACL(static_cast<aclError>(aclnnPidStepResponseFeatures(
            d_pv, d_sp, d_features, data.batch, data.candidates, data.sample_count, data.sample_interval,
            data.settle_band_ratio, workspace, workspace_size, stream)));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        CHECK_ACL(aclrtMemcpy(result.features.data(), features_bytes, d_features, features_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        e2e_total += MsSince(start, Clock::now());
    }
    result.kernel_ms = kernel_total / static_cast<double>(iters);
    result.resident_e2e_ms = e2e_total / static_cast<double>(iters);

    aclrtFree(workspace);
    aclrtFree(d_features);
    aclrtFree(d_sp);
    aclrtFree(d_pv);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const int device_id = argc > 1 ? std::atoi(argv[1]) : 0;
        const int64_t batch = argc > 2 ? std::atoll(argv[2]) : 64;
        const int64_t candidates = argc > 3 ? std::atoll(argv[3]) : 32;
        const int64_t sample_count = argc > 4 ? std::atoll(argv[4]) : 1024;
        const int iters = argc > 5 ? std::atoi(argv[5]) : 5;
        const int cpu_threads = argc > 6 ? std::atoi(argv[6]) : 64;

        CHECK_ACL(aclInit(nullptr));
        const CaseData data = MakeCase(batch, candidates, sample_count);
        const auto cpu = CpuReference(data, cpu_threads);
        const double cpu_ms = MeasureCpuMs(data, cpu_threads, std::max(1, std::min(3, iters)));
        const NpuResult npu = RunNpu(data, device_id, iters);
        const ErrorStats err = Compare(npu.features, cpu);
        CHECK_ACL(aclFinalize());

        std::cout << "B=" << batch << " C=" << candidates << " N=" << sample_count
                  << " cpu_" << cpu_threads << "T_ms=" << cpu_ms << " npu_kernel_ms=" << npu.kernel_ms
                  << " npu_resident_e2e_ms=" << npu.resident_e2e_ms
                  << " kernel_speedup=" << (cpu_ms / npu.kernel_ms)
                  << " resident_e2e_speedup=" << (cpu_ms / npu.resident_e2e_ms)
                  << " feature_max_abs=" << err.max_abs << " feature_max_rel=" << err.max_rel << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "FAILED: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
