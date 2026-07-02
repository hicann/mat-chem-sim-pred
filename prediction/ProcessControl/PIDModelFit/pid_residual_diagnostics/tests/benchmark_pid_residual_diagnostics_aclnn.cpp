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
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "pid_residual_diagnostics_host.h"

namespace {

using Clock = std::chrono::steady_clock;

struct DiagnosticCase {
    std::vector<float> actual;
    std::vector<float> predicted;
    int64_t batch = 0;
    int64_t sample_count = 0;
    int64_t max_lag = 0;
};

struct DiagnosticResult {
    std::vector<float> metrics;
    std::vector<float> autocorr;
};

struct NpuResult {
    DiagnosticResult values;
    double kernel_ms = 0.0;
    double e2e_ms = 0.0;
};

struct ErrorStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t max_index = 0U;
    float lhs = 0.0f;
    float rhs = 0.0f;
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

DiagnosticCase MakeCase(int64_t batch, int64_t sample_count, int64_t max_lag)
{
    DiagnosticCase data;
    data.batch = batch;
    data.sample_count = sample_count;
    data.max_lag = max_lag;
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

void ComputeOne(const DiagnosticCase& data, int64_t b, DiagnosticResult& out)
{
    constexpr float kEps = 1.0e-6f;
    const size_t base = static_cast<size_t>(b * data.sample_count);
    float actual_sum = 0.0f;
    float residual_sum = 0.0f;
    for (int64_t i = 0; i < data.sample_count; ++i) {
        const float actual = data.actual[base + static_cast<size_t>(i)];
        const float residual = actual - data.predicted[base + static_cast<size_t>(i)];
        actual_sum += actual;
        residual_sum += residual;
    }

    const float n = static_cast<float>(data.sample_count);
    const float actual_mean = actual_sum / n;
    const float residual_mean = residual_sum / n;
    float residual_energy = 0.0f;
    float actual_energy = 0.0f;
    float sse = 0.0f;
    float abs_sum = 0.0f;
    float max_abs = 0.0f;
    float diff_sum = 0.0f;
    float prev_residual = 0.0f;
    for (int64_t i = 0; i < data.sample_count; ++i) {
        const float actual = data.actual[base + static_cast<size_t>(i)];
        const float residual = actual - data.predicted[base + static_cast<size_t>(i)];
        residual_energy += (residual - residual_mean) * (residual - residual_mean);
        actual_energy += (actual - actual_mean) * (actual - actual_mean);
        sse += residual * residual;
        abs_sum += std::fabs(residual);
        max_abs = std::max(max_abs, std::fabs(residual));
        if (i > 0) {
            const float diff = residual - prev_residual;
            diff_sum += diff * diff;
        }
        prev_residual = residual;
    }

    float q = 0.0f;
    const size_t autocorr_base = static_cast<size_t>(b * data.max_lag);
    for (int64_t lag = 1; lag <= data.max_lag; ++lag) {
        float numerator = 0.0f;
        for (int64_t i = lag; i < data.sample_count; ++i) {
            const float lhs = data.actual[base + static_cast<size_t>(i)] -
                              data.predicted[base + static_cast<size_t>(i)] - residual_mean;
            const float rhs = data.actual[base + static_cast<size_t>(i - lag)] -
                              data.predicted[base + static_cast<size_t>(i - lag)] - residual_mean;
            numerator += lhs * rhs;
        }
        const float value = numerator / std::max(residual_energy, kEps);
        out.autocorr[autocorr_base + static_cast<size_t>(lag - 1)] = value;
        q += value * value / std::max(static_cast<float>(data.sample_count - lag), 1.0f);
    }
    q *= n * (n + 2.0f);

    const size_t out_base = static_cast<size_t>(b * PID_RESIDUAL_DIAGNOSTICS_METRIC_COUNT);
    out.metrics[out_base + 0U] = residual_mean;
    out.metrics[out_base + 1U] = std::sqrt(std::max(residual_energy / static_cast<float>(data.sample_count - 1), 0.0f));
    out.metrics[out_base + 2U] = abs_sum / n;
    out.metrics[out_base + 3U] = std::sqrt(sse / n);
    out.metrics[out_base + 4U] = max_abs;
    out.metrics[out_base + 5U] = 100.0f * (1.0f - std::sqrt(sse) / std::sqrt(std::max(actual_energy, kEps)));
    out.metrics[out_base + 6U] = diff_sum / std::max(sse, kEps);
    out.metrics[out_base + 7U] = q;
}

void ComputeRange(const DiagnosticCase& data, int64_t begin, int64_t end, DiagnosticResult& out)
{
    for (int64_t b = begin; b < end; ++b) {
        ComputeOne(data, b, out);
    }
}

DiagnosticResult CpuDiagnostics(const DiagnosticCase& data)
{
    DiagnosticResult out;
    out.metrics.assign(static_cast<size_t>(data.batch * PID_RESIDUAL_DIAGNOSTICS_METRIC_COUNT), 0.0f);
    out.autocorr.assign(static_cast<size_t>(data.batch * data.max_lag), 0.0f);
    ComputeRange(data, 0, data.batch, out);
    return out;
}

double MeasureCpuMs(const DiagnosticCase& data, int threads, int iters, const char* guard_label)
{
    const int worker_count = static_cast<int>(std::min<int64_t>(std::max(1, threads), data.batch));
    DiagnosticResult out;
    out.metrics.assign(static_cast<size_t>(data.batch * PID_RESIDUAL_DIAGNOSTICS_METRIC_COUNT), 0.0f);
    out.autocorr.assign(static_cast<size_t>(data.batch * data.max_lag), 0.0f);
    volatile double guard = 0.0;
    if (worker_count <= 1) {
        auto start = Clock::now();
        for (int i = 0; i < iters; ++i) {
            ComputeRange(data, 0, data.batch, out);
            guard += out.metrics[static_cast<size_t>(i) % out.metrics.size()];
        }
        auto end = Clock::now();
        std::cout << guard_label << "=" << guard << std::endl;
        return MsSince(start, end) / static_cast<double>(iters);
    }

    std::mutex mutex;
    std::condition_variable start_cv;
    std::condition_variable done_cv;
    int epoch = 0;
    int completed = 0;
    bool stop = false;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int worker = 0; worker < worker_count; ++worker) {
        const int64_t begin = data.batch * worker / worker_count;
        const int64_t end = data.batch * (worker + 1) / worker_count;
        workers.emplace_back([&, begin, end]() {
            int seen_epoch = 0;
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    start_cv.wait(lock, [&]() { return stop || epoch != seen_epoch; });
                    if (stop) {
                        return;
                    }
                    seen_epoch = epoch;
                }
                ComputeRange(data, begin, end, out);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    ++completed;
                    if (completed == worker_count) {
                        done_cv.notify_one();
                    }
                }
            }
        });
    }

    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            completed = 0;
            ++epoch;
        }
        start_cv.notify_all();
        {
            std::unique_lock<std::mutex> lock(mutex);
            done_cv.wait(lock, [&]() { return completed == worker_count; });
        }
        guard += out.metrics[static_cast<size_t>(i) % out.metrics.size()];
    }
    auto end = Clock::now();
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
        ++epoch;
    }
    start_cv.notify_all();
    for (auto& worker : workers) {
        worker.join();
    }
    std::cout << guard_label << "=" << guard << std::endl;
    return MsSince(start, end) / static_cast<double>(iters);
}

void* MallocDevice(size_t bytes)
{
    void* ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

NpuResult RunNpu(const DiagnosticCase& data, int device_id, int iters, int warmup)
{
    NpuResult result;
    result.values.metrics.assign(static_cast<size_t>(data.batch * PID_RESIDUAL_DIAGNOSTICS_METRIC_COUNT), 0.0f);
    result.values.autocorr.assign(static_cast<size_t>(data.batch * data.max_lag), 0.0f);
    aclrtStream stream = nullptr;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    CHECK_ACL(aclrtCreateStream(&stream));
    void* d_actual = MallocDevice(data.actual.size() * sizeof(float));
    void* d_predicted = MallocDevice(data.predicted.size() * sizeof(float));
    void* d_metrics = MallocDevice(result.values.metrics.size() * sizeof(float));
    void* d_autocorr = MallocDevice(result.values.autocorr.size() * sizeof(float));
    const uint64_t workspace_size =
        aclnnPidResidualDiagnosticsGetWorkspaceSize(data.batch, data.sample_count, data.max_lag);
    void* workspace = MallocDevice(workspace_size);

    auto copy_inputs = [&]() {
        CHECK_ACL(aclrtMemcpy(d_actual, data.actual.size() * sizeof(float), data.actual.data(),
                              data.actual.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_predicted, data.predicted.size() * sizeof(float), data.predicted.data(),
                              data.predicted.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    };
    auto run_once = [&]() {
        const int32_t ret = aclnnPidResidualDiagnostics(
            d_actual, d_predicted, d_metrics, d_autocorr, data.batch, data.sample_count, data.max_lag, workspace,
            workspace_size, stream);
        CHECK_ACL(static_cast<aclError>(ret));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    };
    copy_inputs();
    for (int i = 0; i < warmup; ++i) {
        run_once();
    }

    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        run_once();
    }
    auto end = Clock::now();
    result.kernel_ms = MsSince(start, end) / static_cast<double>(iters);

    start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        copy_inputs();
        run_once();
        CHECK_ACL(aclrtMemcpy(result.values.metrics.data(), result.values.metrics.size() * sizeof(float), d_metrics,
                              result.values.metrics.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(result.values.autocorr.data(), result.values.autocorr.size() * sizeof(float), d_autocorr,
                              result.values.autocorr.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    }
    end = Clock::now();
    result.e2e_ms = MsSince(start, end) / static_cast<double>(iters);

    CHECK_ACL(aclrtMemcpy(result.values.metrics.data(), result.values.metrics.size() * sizeof(float), d_metrics,
                          result.values.metrics.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(result.values.autocorr.data(), result.values.autocorr.size() * sizeof(float), d_autocorr,
                          result.values.autocorr.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    aclrtFree(workspace);
    aclrtFree(d_autocorr);
    aclrtFree(d_metrics);
    aclrtFree(d_predicted);
    aclrtFree(d_actual);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return result;
}

ErrorStats ComputeErrorStats(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    ErrorStats stats;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float abs_err = std::fabs(lhs[i] - rhs[i]);
        const float rel_err = abs_err / std::max(std::fabs(lhs[i]), 1.0f);
        if (abs_err > stats.max_abs) {
            stats.max_abs = abs_err;
            stats.max_rel = rel_err;
            stats.max_index = i;
            stats.lhs = lhs[i];
            stats.rhs = rhs[i];
        }
    }
    return stats;
}

int ResolveThreadCount(int requested)
{
    if (requested > 0) {
        return requested;
    }
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    return hardware_threads == 0 ? 1 : static_cast<int>(hardware_threads);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <device_id> [batch=256] [sample_count=2048] [max_lag=32] [iters=10] [warmup=2]"
                     " [cpu_threads=hardware]"
                  << std::endl;
        return 1;
    }
    const int device_id = std::atoi(argv[1]);
    const int64_t batch = argc > 2 ? std::atoll(argv[2]) : 256;
    const int64_t sample_count = argc > 3 ? std::atoll(argv[3]) : 2048;
    const int64_t max_lag = argc > 4 ? std::atoll(argv[4]) : 32;
    const int iters = argc > 5 ? std::atoi(argv[5]) : 10;
    const int warmup = argc > 6 ? std::atoi(argv[6]) : 2;
    const int cpu_threads = ResolveThreadCount(argc > 7 ? std::atoi(argv[7]) : 0);
    if (batch <= 0 || sample_count <= 1 || max_lag <= 0 || max_lag >= sample_count || iters <= 0 || warmup < 0) {
        std::cerr << "invalid benchmark arguments" << std::endl;
        return 1;
    }

    try {
        const DiagnosticCase data = MakeCase(batch, sample_count, max_lag);
        const DiagnosticResult cpu = CpuDiagnostics(data);
        const double cpu_single_ms =
            MeasureCpuMs(data, 1, std::max(1, std::min(3, iters)), "cpu_single_guard");
        const double cpu_parallel_ms = MeasureCpuMs(data, cpu_threads, iters, "cpu_parallel_guard");
        const NpuResult npu = RunNpu(data, device_id, iters, warmup);
        const ErrorStats metric_err = ComputeErrorStats(cpu.metrics, npu.values.metrics);
        const ErrorStats autocorr_err = ComputeErrorStats(cpu.autocorr, npu.values.autocorr);

        std::cout << "op=PidResidualDiagnostics batch=" << batch << " sample_count=" << sample_count
                  << " max_lag=" << max_lag << " iters=" << iters << " warmup=" << warmup << std::endl;
        std::cout << "cpu_threads=" << cpu_threads << std::endl;
        std::cout << "cpu_single_ms_avg=" << cpu_single_ms << std::endl;
        std::cout << "cpu_parallel_ms_avg=" << cpu_parallel_ms << std::endl;
        std::cout << "npu_kernel_ms_avg=" << npu.kernel_ms << std::endl;
        std::cout << "npu_e2e_ms_avg=" << npu.e2e_ms << std::endl;
        std::cout << "speedup_npu_kernel_vs_cpu_parallel=" << (cpu_parallel_ms / npu.kernel_ms) << std::endl;
        std::cout << "speedup_npu_e2e_vs_cpu_parallel=" << (cpu_parallel_ms / npu.e2e_ms) << std::endl;
        std::cout << "speedup_npu_kernel_vs_cpu_single=" << (cpu_single_ms / npu.kernel_ms) << std::endl;
        std::cout << "max_metric_abs_err=" << metric_err.max_abs << std::endl;
        std::cout << "max_metric_rel_err=" << metric_err.max_rel << std::endl;
        std::cout << "max_metric_err_metric_index="
                  << (metric_err.max_index % PID_RESIDUAL_DIAGNOSTICS_METRIC_COUNT) << std::endl;
        std::cout << "max_metric_err_cpu=" << metric_err.lhs << std::endl;
        std::cout << "max_metric_err_npu=" << metric_err.rhs << std::endl;
        std::cout << "max_autocorr_abs_err=" << autocorr_err.max_abs << std::endl;
        std::cout << "max_autocorr_rel_err=" << autocorr_err.max_rel << std::endl;
        std::cout << "max_autocorr_err_lag_index=" << (autocorr_err.max_index % static_cast<size_t>(max_lag))
                  << std::endl;
        return (metric_err.max_abs < 2.0e-2f && autocorr_err.max_abs < 2.0e-4f) ? 0 : 2;
    } catch (const std::exception& ex) {
        std::cerr << "benchmark failed: " << ex.what() << std::endl;
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }
}
