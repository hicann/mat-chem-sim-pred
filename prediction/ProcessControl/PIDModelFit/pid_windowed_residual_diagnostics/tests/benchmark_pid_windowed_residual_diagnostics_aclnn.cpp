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
#include "pid_windowed_residual_diagnostics_host.h"

namespace {

using Clock = std::chrono::steady_clock;

struct CaseData {
    std::vector<float> actual;
    std::vector<float> predicted;
    int64_t batch = 0;
    int64_t sample_count = 0;
    int64_t window_size = 0;
    int64_t stride = 0;
    int64_t max_lag = 0;
    int64_t window_count = 0;
};

struct ResultData {
    std::vector<float> metrics;
    std::vector<float> autocorr;
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

void* MallocDevice(size_t bytes)
{
    void* ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

CaseData MakeCase(int64_t batch, int64_t sample_count, int64_t window_size, int64_t stride, int64_t max_lag)
{
    CaseData data;
    data.batch = batch;
    data.sample_count = sample_count;
    data.window_size = window_size;
    data.stride = stride;
    data.max_lag = max_lag;
    data.window_count = aclnnPidWindowedResidualDiagnosticsGetWindowCount(sample_count, window_size, stride);
    data.actual.assign(static_cast<size_t>(batch * sample_count), 0.0f);
    data.predicted.assign(static_cast<size_t>(batch * sample_count), 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        const float bias = -0.08f + 0.16f * static_cast<float>(b) / static_cast<float>(std::max<int64_t>(1, batch - 1));
        for (int64_t i = 0; i < sample_count; ++i) {
            const float phase = static_cast<float>((i * 17 + b * 13) % 4096) / 4096.0f;
            const float actual = 50.0f + 2.0f * std::sin(6.283185307179586f * phase) +
                                 0.35f * std::cos(18.84955592153876f * phase + static_cast<float>(b) * 0.017f);
            const float residual = bias +
                                   0.08f * std::sin(6.283185307179586f *
                                                     (phase * (2.0f + static_cast<float>(b % 5) * 0.04f) +
                                                      static_cast<float>(b) * 0.011f));
            const size_t offset = static_cast<size_t>(b * sample_count + i);
            data.actual[offset] = actual;
            data.predicted[offset] = actual - residual;
        }
    }
    return data;
}

void ComputeTask(const CaseData& data, int64_t task, ResultData& out)
{
    constexpr float kEps = 1.0e-6f;
    const int64_t b = task / data.window_count;
    const int64_t window = task - b * data.window_count;
    const int64_t start = window * data.stride;
    const size_t base = static_cast<size_t>(b * data.sample_count + start);

    float actual_sum = 0.0f;
    float residual_sum = 0.0f;
    for (int64_t i = 0; i < data.window_size; ++i) {
        const float actual = data.actual[base + static_cast<size_t>(i)];
        actual_sum += actual;
        residual_sum += actual - data.predicted[base + static_cast<size_t>(i)];
    }

    const float n = static_cast<float>(data.window_size);
    const float actual_mean = actual_sum / n;
    const float residual_mean = residual_sum / n;
    float residual_energy = 0.0f;
    float actual_energy = 0.0f;
    float sse = 0.0f;
    float abs_sum = 0.0f;
    float max_abs = 0.0f;
    float diff_sum = 0.0f;
    float previous_residual = 0.0f;
    for (int64_t i = 0; i < data.window_size; ++i) {
        const float actual = data.actual[base + static_cast<size_t>(i)];
        const float residual = actual - data.predicted[base + static_cast<size_t>(i)];
        residual_energy += (residual - residual_mean) * (residual - residual_mean);
        actual_energy += (actual - actual_mean) * (actual - actual_mean);
        sse += residual * residual;
        abs_sum += std::fabs(residual);
        max_abs = std::max(max_abs, std::fabs(residual));
        if (i > 0) {
            const float diff = residual - previous_residual;
            diff_sum += diff * diff;
        }
        previous_residual = residual;
    }

    float q = 0.0f;
    const size_t autocorr_base = static_cast<size_t>(task * data.max_lag);
    for (int64_t lag = 1; lag <= data.max_lag; ++lag) {
        float numerator = 0.0f;
        for (int64_t i = lag; i < data.window_size; ++i) {
            const float lhs = data.actual[base + static_cast<size_t>(i)] -
                              data.predicted[base + static_cast<size_t>(i)] - residual_mean;
            const float rhs = data.actual[base + static_cast<size_t>(i - lag)] -
                              data.predicted[base + static_cast<size_t>(i - lag)] - residual_mean;
            numerator += lhs * rhs;
        }
        const float value = numerator / std::max(residual_energy, kEps);
        out.autocorr[autocorr_base + static_cast<size_t>(lag - 1)] = value;
        q += value * value / std::max(static_cast<float>(data.window_size - lag), 1.0f);
    }
    q *= n * (n + 2.0f);

    const size_t out_base = static_cast<size_t>(task * PID_WINDOWED_RESIDUAL_DIAGNOSTICS_METRIC_COUNT);
    out.metrics[out_base + 0U] = residual_mean;
    out.metrics[out_base + 1U] = std::sqrt(std::max(residual_energy / std::max(n - 1.0f, 1.0f), 0.0f));
    out.metrics[out_base + 2U] = abs_sum / n;
    out.metrics[out_base + 3U] = std::sqrt(sse / n);
    out.metrics[out_base + 4U] = max_abs;
    out.metrics[out_base + 5U] = 100.0f * (1.0f - std::sqrt(sse) / std::sqrt(std::max(actual_energy, kEps)));
    out.metrics[out_base + 6U] = diff_sum / std::max(sse, kEps);
    out.metrics[out_base + 7U] = q;
}

ResultData CpuReference(const CaseData& data, int threads)
{
    const int64_t total_tasks = data.batch * data.window_count;
    const int worker_count = static_cast<int>(std::min<int64_t>(std::max(1, threads), total_tasks));
    ResultData out;
    out.metrics.assign(static_cast<size_t>(total_tasks * PID_WINDOWED_RESIDUAL_DIAGNOSTICS_METRIC_COUNT), 0.0f);
    out.autocorr.assign(static_cast<size_t>(total_tasks * data.max_lag), 0.0f);

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
        ResultData out = CpuReference(data, threads);
        guard += out.metrics[static_cast<size_t>(i) % out.metrics.size()];
    }
    const auto end = Clock::now();
    std::cout << "cpu_guard=" << guard << std::endl;
    return MsSince(start, end) / static_cast<double>(iters);
}

ErrorStats Compare(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    ErrorStats stats;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float abs_err = std::fabs(lhs[i] - rhs[i]);
        const float rel_err = abs_err / std::max(std::fabs(rhs[i]), 1.0e-6f);
        stats.max_abs = std::max(stats.max_abs, abs_err);
        stats.max_rel = std::max(stats.max_rel, rel_err);
    }
    return stats;
}

ResultData RunNpu(
    const CaseData& data, int device_id, int iters, double& kernel_ms, double& resident_e2e_ms)
{
    const size_t input_bytes = data.actual.size() * sizeof(float);
    const size_t metrics_bytes =
        static_cast<size_t>(data.batch * data.window_count * PID_WINDOWED_RESIDUAL_DIAGNOSTICS_METRIC_COUNT) *
        sizeof(float);
    const size_t autocorr_bytes = static_cast<size_t>(data.batch * data.window_count * data.max_lag) * sizeof(float);

    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    void* d_actual = MallocDevice(input_bytes);
    void* d_predicted = MallocDevice(input_bytes);
    void* d_metrics = MallocDevice(metrics_bytes);
    void* d_autocorr = MallocDevice(autocorr_bytes);
    const uint64_t workspace_size = aclnnPidWindowedResidualDiagnosticsGetWorkspaceSize(
        data.batch, data.sample_count, data.window_size, data.stride, data.max_lag);
    void* workspace = MallocDevice(workspace_size);

    CHECK_ACL(aclrtMemcpy(d_actual, input_bytes, data.actual.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_predicted, input_bytes, data.predicted.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(static_cast<aclError>(aclnnPidWindowedResidualDiagnostics(
        d_actual, d_predicted, d_metrics, d_autocorr, data.batch, data.sample_count, data.window_size, data.stride,
        data.max_lag, workspace, workspace_size, stream)));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    double kernel_total = 0.0;
    for (int i = 0; i < iters; ++i) {
        const auto start = Clock::now();
        CHECK_ACL(static_cast<aclError>(aclnnPidWindowedResidualDiagnostics(
            d_actual, d_predicted, d_metrics, d_autocorr, data.batch, data.sample_count, data.window_size, data.stride,
            data.max_lag, workspace, workspace_size, stream)));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        kernel_total += MsSince(start, Clock::now());
    }
    kernel_ms = kernel_total / static_cast<double>(iters);

    ResultData out;
    out.metrics.assign(metrics_bytes / sizeof(float), 0.0f);
    out.autocorr.assign(autocorr_bytes / sizeof(float), 0.0f);
    double e2e_total = 0.0;
    for (int i = 0; i < iters; ++i) {
        const auto e2e_start = Clock::now();
        CHECK_ACL(static_cast<aclError>(aclnnPidWindowedResidualDiagnostics(
            d_actual, d_predicted, d_metrics, d_autocorr, data.batch, data.sample_count, data.window_size,
            data.stride, data.max_lag, workspace, workspace_size, stream)));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        CHECK_ACL(aclrtMemcpy(out.metrics.data(), metrics_bytes, d_metrics, metrics_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(
            aclrtMemcpy(out.autocorr.data(), autocorr_bytes, d_autocorr, autocorr_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        e2e_total += MsSince(e2e_start, Clock::now());
    }
    resident_e2e_ms = e2e_total / static_cast<double>(iters);

    aclrtFree(workspace);
    aclrtFree(d_autocorr);
    aclrtFree(d_metrics);
    aclrtFree(d_predicted);
    aclrtFree(d_actual);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    return out;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const int device_id = argc > 1 ? std::atoi(argv[1]) : 0;
        const int64_t batch = argc > 2 ? std::atoll(argv[2]) : 128;
        const int64_t sample_count = argc > 3 ? std::atoll(argv[3]) : 4096;
        const int64_t window_size = argc > 4 ? std::atoll(argv[4]) : 512;
        const int64_t stride = argc > 5 ? std::atoll(argv[5]) : 256;
        const int64_t max_lag = argc > 6 ? std::atoll(argv[6]) : 32;
        const int iters = argc > 7 ? std::atoi(argv[7]) : 5;
        const int cpu_threads = argc > 8 ? std::atoi(argv[8]) : 64;

        CHECK_ACL(aclInit(nullptr));
        const CaseData data = MakeCase(batch, sample_count, window_size, stride, max_lag);
        const ResultData cpu = CpuReference(data, cpu_threads);
        const double cpu_ms = MeasureCpuMs(data, cpu_threads, std::max(1, std::min(3, iters)));

        double kernel_ms = 0.0;
        double resident_e2e_ms = 0.0;
        const ResultData npu = RunNpu(data, device_id, iters, kernel_ms, resident_e2e_ms);
        const ErrorStats metric_err = Compare(npu.metrics, cpu.metrics);
        const ErrorStats autocorr_err = Compare(npu.autocorr, cpu.autocorr);
        CHECK_ACL(aclFinalize());

        std::cout << "B=" << batch << " N=" << sample_count << " windows=" << data.window_count
                  << " window=" << window_size << " stride=" << stride << " lag=" << max_lag
                  << " cpu_" << cpu_threads << "T_ms=" << cpu_ms << " npu_kernel_ms=" << kernel_ms
                  << " npu_resident_e2e_ms=" << resident_e2e_ms << " kernel_speedup=" << (cpu_ms / kernel_ms)
                  << " resident_e2e_speedup=" << (cpu_ms / resident_e2e_ms) << " metric_max_abs=" << metric_err.max_abs
                  << " metric_max_rel=" << metric_err.max_rel << " autocorr_max_abs=" << autocorr_err.max_abs
                  << " autocorr_max_rel=" << autocorr_err.max_rel << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "FAILED: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
