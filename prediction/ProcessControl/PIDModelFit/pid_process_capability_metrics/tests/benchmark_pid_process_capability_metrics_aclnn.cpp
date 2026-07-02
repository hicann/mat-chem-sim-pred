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
#include "pid_process_capability_metrics_host.h"

namespace {

using Clock = std::chrono::steady_clock;

struct TensorHandle {
    void* device = nullptr;
};

struct CapabilityCase {
    std::vector<float> values;
    std::vector<float> lsl;
    std::vector<float> usl;
    int64_t batch = 0;
    int64_t sample_count = 0;
};

struct NpuResult {
    std::vector<float> metrics;
    double kernel_ms = 0.0;
    double e2e_ms = 0.0;
};

using ComputeRangeFn = void (*)(const CapabilityCase&, int64_t, int64_t, std::vector<float>&);

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

CapabilityCase MakeCase(int64_t batch, int64_t sample_count)
{
    CapabilityCase data;
    data.batch = batch;
    data.sample_count = sample_count;
    data.values.assign(static_cast<size_t>(batch * sample_count), 0.0f);
    data.lsl.assign(static_cast<size_t>(batch), 47.0f);
    data.usl.assign(static_cast<size_t>(batch), 53.0f);
    for (int64_t b = 0; b < batch; ++b) {
        const float drift = -0.4f + 0.8f * static_cast<float>(b) / static_cast<float>(std::max<int64_t>(1, batch - 1));
        const float scale = 0.8f + 0.004f * static_cast<float>(b % 37);
        for (int64_t i = 0; i < sample_count; ++i) {
            const float phase = static_cast<float>((i * 17 + b * 13) % 1024) / 1024.0f;
            const float wave = std::sin(phase * 6.283185307179586f) + 0.25f * std::cos(phase * 18.84955592153876f);
            data.values[static_cast<size_t>(b * sample_count + i)] = 50.0f + drift + scale * wave;
        }
    }
    if (batch > 0 && sample_count > 1) {
        data.values[0] = 53.5f;
        data.values[1] = 46.5f;
        data.values[data.values.size() - 1U] = 54.0f;
    }
    return data;
}

void StoreMetrics(
    const CapabilityCase& data, int64_t b, double mean, double var_population, double var_sample,
    int64_t out_count, float min_value, float max_value, std::vector<float>& metrics)
{
    constexpr float kEps = 1.0e-6f;
    const double n = static_cast<double>(data.sample_count);
    const double std_population = std::sqrt(std::max(0.0, var_population));
    const double std_sample = std::sqrt(std::max(0.0, var_sample));
    const double denom_sample = 3.0 * std::max(std_sample, static_cast<double>(kEps));
    const double denom_population = 3.0 * std::max(std_population, static_cast<double>(kEps));
    const double cpu = (data.usl[static_cast<size_t>(b)] - mean) / denom_sample;
    const double cpl = (mean - data.lsl[static_cast<size_t>(b)]) / denom_sample;
    const double ppu = (data.usl[static_cast<size_t>(b)] - mean) / denom_population;
    const double ppl = (mean - data.lsl[static_cast<size_t>(b)]) / denom_population;
    const double spec_width = data.usl[static_cast<size_t>(b)] - data.lsl[static_cast<size_t>(b)];
    const size_t out = static_cast<size_t>(b * PID_PROCESS_CAPABILITY_METRIC_COUNT);
    metrics[out + 0U] = static_cast<float>(mean);
    metrics[out + 1U] = static_cast<float>(std_sample);
    metrics[out + 2U] = static_cast<float>(std_population);
    metrics[out + 3U] = static_cast<float>(spec_width / (2.0 * denom_sample));
    metrics[out + 4U] = static_cast<float>(cpu);
    metrics[out + 5U] = static_cast<float>(cpl);
    metrics[out + 6U] = static_cast<float>(std::min(cpu, cpl));
    metrics[out + 7U] = static_cast<float>(spec_width / (2.0 * denom_population));
    metrics[out + 8U] = static_cast<float>(std::min(ppu, ppl));
    metrics[out + 9U] = static_cast<float>(static_cast<double>(out_count) / n);
    metrics[out + 10U] = static_cast<float>(out_count);
    metrics[out + 11U] = min_value;
    metrics[out + 12U] = max_value;
}

void ComputeRangeStable(const CapabilityCase& data, int64_t begin, int64_t end, std::vector<float>& metrics)
{
    for (int64_t b = begin; b < end; ++b) {
        double mean = 0.0;
        double m2 = 0.0;
        int64_t out_count = 0;
        float min_value = data.values[static_cast<size_t>(b * data.sample_count)];
        float max_value = min_value;
        for (int64_t i = 0; i < data.sample_count; ++i) {
            const float value = data.values[static_cast<size_t>(b * data.sample_count + i)];
            const double count = static_cast<double>(i + 1);
            const double delta = static_cast<double>(value) - mean;
            mean += delta / count;
            const double delta2 = static_cast<double>(value) - mean;
            m2 += delta * delta2;
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            if (value < data.lsl[static_cast<size_t>(b)] || value > data.usl[static_cast<size_t>(b)]) {
                ++out_count;
            }
        }
        const double n = static_cast<double>(data.sample_count);
        const double var_population = std::max(0.0, m2 / n);
        const double var_sample =
            data.sample_count > 1 ? m2 / static_cast<double>(data.sample_count - 1) : var_population;
        StoreMetrics(data, b, mean, var_population, var_sample, out_count, min_value, max_value, metrics);
    }
}

void ComputeRangeFast(const CapabilityCase& data, int64_t begin, int64_t end, std::vector<float>& metrics)
{
    for (int64_t b = begin; b < end; ++b) {
        double sum = 0.0;
        double sum_sq = 0.0;
        int64_t out_count = 0;
        float min_value = data.values[static_cast<size_t>(b * data.sample_count)];
        float max_value = min_value;
        for (int64_t i = 0; i < data.sample_count; ++i) {
            const float value = data.values[static_cast<size_t>(b * data.sample_count + i)];
            sum += static_cast<double>(value);
            sum_sq += static_cast<double>(value) * static_cast<double>(value);
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            if (value < data.lsl[static_cast<size_t>(b)] || value > data.usl[static_cast<size_t>(b)]) {
                ++out_count;
            }
        }
        const double n = static_cast<double>(data.sample_count);
        const double mean = sum / n;
        const double var_population = std::max(0.0, sum_sq / n - mean * mean);
        const double var_sample =
            data.sample_count > 1 ? var_population * n / static_cast<double>(data.sample_count - 1) : var_population;
        StoreMetrics(data, b, mean, var_population, var_sample, out_count, min_value, max_value, metrics);
    }
}

std::vector<float> CpuMetrics(const CapabilityCase& data)
{
    std::vector<float> metrics(static_cast<size_t>(data.batch * PID_PROCESS_CAPABILITY_METRIC_COUNT), 0.0f);
    ComputeRangeStable(data, 0, data.batch, metrics);
    return metrics;
}

std::vector<float> CpuMetricsParallel(const CapabilityCase& data, int threads, ComputeRangeFn compute_range)
{
    if (threads <= 1) {
        std::vector<float> metrics(static_cast<size_t>(data.batch * PID_PROCESS_CAPABILITY_METRIC_COUNT), 0.0f);
        compute_range(data, 0, data.batch, metrics);
        return metrics;
    }
    const int worker_count = static_cast<int>(std::min<int64_t>(threads, data.batch));
    std::vector<float> metrics(static_cast<size_t>(data.batch * PID_PROCESS_CAPABILITY_METRIC_COUNT), 0.0f);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int worker = 0; worker < worker_count; ++worker) {
        const int64_t begin = data.batch * worker / worker_count;
        const int64_t end = data.batch * (worker + 1) / worker_count;
        workers.emplace_back([&data, begin, end, &metrics, compute_range]() {
            compute_range(data, begin, end, metrics);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return metrics;
}

double MeasureCpuMs(
    const CapabilityCase& data, int threads, int iters, ComputeRangeFn compute_range, const char* guard_label)
{
    const int worker_count = static_cast<int>(std::min<int64_t>(std::max(1, threads), data.batch));
    std::vector<float> metrics(static_cast<size_t>(data.batch * PID_PROCESS_CAPABILITY_METRIC_COUNT), 0.0f);
    volatile double guard = 0.0;
    if (worker_count <= 1) {
        auto start = Clock::now();
        for (int i = 0; i < iters; ++i) {
            compute_range(data, 0, data.batch, metrics);
            guard += metrics[static_cast<size_t>(i) % metrics.size()];
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
                compute_range(data, begin, end, metrics);
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
        guard += metrics[static_cast<size_t>(i) % metrics.size()];
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

NpuResult RunNpu(const CapabilityCase& data, int device_id, int iters, int warmup)
{
    NpuResult result;
    result.metrics.assign(static_cast<size_t>(data.batch * PID_PROCESS_CAPABILITY_METRIC_COUNT), 0.0f);
    aclrtStream stream = nullptr;
    void* d_values = nullptr;
    void* d_lsl = nullptr;
    void* d_usl = nullptr;
    void* d_metrics = nullptr;
    void* workspace = nullptr;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    CHECK_ACL(aclrtCreateStream(&stream));
    d_values = MallocDevice(data.values.size() * sizeof(float));
    d_lsl = MallocDevice(data.lsl.size() * sizeof(float));
    d_usl = MallocDevice(data.usl.size() * sizeof(float));
    d_metrics = MallocDevice(result.metrics.size() * sizeof(float));
    const uint64_t workspace_size =
        aclnnPidProcessCapabilityMetricsGetWorkspaceSize(data.batch, data.sample_count);
    workspace = MallocDevice(workspace_size);

    auto copy_inputs = [&]() {
        CHECK_ACL(aclrtMemcpy(d_values, data.values.size() * sizeof(float), data.values.data(),
                              data.values.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_lsl, data.lsl.size() * sizeof(float), data.lsl.data(), data.lsl.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_usl, data.usl.size() * sizeof(float), data.usl.data(), data.usl.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
    };
    auto run_once = [&]() {
        const int32_t ret = aclnnPidProcessCapabilityMetrics(
            d_values, d_lsl, d_usl, d_metrics, data.batch, data.sample_count, workspace, workspace_size, stream);
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
        CHECK_ACL(aclrtMemcpy(result.metrics.data(), result.metrics.size() * sizeof(float), d_metrics,
                              result.metrics.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    }
    end = Clock::now();
    result.e2e_ms = MsSince(start, end) / static_cast<double>(iters);

    CHECK_ACL(aclrtMemcpy(result.metrics.data(), result.metrics.size() * sizeof(float), d_metrics,
                          result.metrics.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    aclrtFree(workspace);
    aclrtFree(d_metrics);
    aclrtFree(d_usl);
    aclrtFree(d_lsl);
    aclrtFree(d_values);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return result;
}

float MaxAbsErr(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    float err = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        err = std::max(err, std::fabs(lhs[i] - rhs[i]));
    }
    return err;
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
                  << " <device_id> [batch=128] [sample_count=4096] [iters=20] [warmup=3] [cpu_threads=hardware]"
                  << std::endl;
        return 1;
    }
    const int device_id = std::atoi(argv[1]);
    const int64_t batch = argc > 2 ? std::atoll(argv[2]) : 128;
    const int64_t sample_count = argc > 3 ? std::atoll(argv[3]) : 4096;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 20;
    const int warmup = argc > 5 ? std::atoi(argv[5]) : 3;
    const int cpu_threads = ResolveThreadCount(argc > 6 ? std::atoi(argv[6]) : 0);
    if (batch <= 0 || sample_count <= 1 || iters <= 0 || warmup < 0) {
        std::cerr << "batch and iters must be positive; sample_count must be > 1; warmup must be non-negative"
                  << std::endl;
        return 1;
    }

    try {
        const CapabilityCase data = MakeCase(batch, sample_count);
        const std::vector<float> cpu = CpuMetrics(data);
        const double cpu_single_ms =
            MeasureCpuMs(data, 1, std::max(1, std::min(3, iters)), ComputeRangeStable, "cpu_stable_single_guard");
        const double cpu_parallel_ms =
            MeasureCpuMs(data, cpu_threads, iters, ComputeRangeStable, "cpu_stable_parallel_guard");
        const double cpu_fast_single_ms =
            MeasureCpuMs(data, 1, std::max(1, std::min(3, iters)), ComputeRangeFast, "cpu_fast_single_guard");
        const double cpu_fast_parallel_ms =
            MeasureCpuMs(data, cpu_threads, iters, ComputeRangeFast, "cpu_fast_parallel_guard");
        const std::vector<float> cpu_fast = CpuMetricsParallel(data, cpu_threads, ComputeRangeFast);
        const NpuResult npu = RunNpu(data, device_id, iters, warmup);
        const float max_err = MaxAbsErr(cpu, npu.metrics);
        const float max_err_fast_cpu = MaxAbsErr(cpu, cpu_fast);

        std::cout << "op=PidProcessCapabilityMetrics batch=" << batch << " sample_count=" << sample_count
                  << " iters=" << iters << " warmup=" << warmup << std::endl;
        std::cout << "cpu_threads=" << cpu_threads << std::endl;
        std::cout << "cpu_single_ms_avg=" << cpu_single_ms << std::endl;
        std::cout << "cpu_parallel_ms_avg=" << cpu_parallel_ms << std::endl;
        std::cout << "cpu_fast_single_ms_avg=" << cpu_fast_single_ms << std::endl;
        std::cout << "cpu_fast_parallel_ms_avg=" << cpu_fast_parallel_ms << std::endl;
        std::cout << "npu_kernel_ms_avg=" << npu.kernel_ms << std::endl;
        std::cout << "npu_e2e_ms_avg=" << npu.e2e_ms << std::endl;
        std::cout << "speedup_npu_kernel_vs_cpu_parallel=" << (cpu_parallel_ms / npu.kernel_ms) << std::endl;
        std::cout << "speedup_npu_e2e_vs_cpu_parallel=" << (cpu_parallel_ms / npu.e2e_ms) << std::endl;
        std::cout << "speedup_npu_kernel_vs_cpu_fast_parallel=" << (cpu_fast_parallel_ms / npu.kernel_ms)
                  << std::endl;
        std::cout << "speedup_npu_e2e_vs_cpu_fast_parallel=" << (cpu_fast_parallel_ms / npu.e2e_ms) << std::endl;
        std::cout << "speedup_npu_kernel_vs_cpu_single=" << (cpu_single_ms / npu.kernel_ms) << std::endl;
        std::cout << "max_abs_err=" << max_err << std::endl;
        std::cout << "max_abs_err_fast_cpu_vs_stable_cpu=" << max_err_fast_cpu << std::endl;
        std::cout << "metric_order=mean,std_sample,std_population,cp,cpu,cpl,cpk,pp,ppk,out_ratio,out_count,min,max"
                  << std::endl;
        return max_err < 5.0e-2f ? 0 : 2;
    } catch (const std::exception& ex) {
        std::cerr << "benchmark failed: " << ex.what() << std::endl;
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }
}
