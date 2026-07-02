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
#include "pid_control_performance_metrics_host.h"

namespace {

using Clock = std::chrono::steady_clock;

struct PerformanceCase {
    std::vector<float> pv;
    std::vector<float> sp;
    std::vector<float> lsl;
    std::vector<float> usl;
    std::vector<float> mv_variance;
    int64_t batch = 0;
    int64_t sample_count = 0;
    float sample_interval = 1.0f;
    float settle_band = 0.1f;
};

struct NpuResult {
    std::vector<float> metrics;
    double kernel_ms = 0.0;
    double e2e_ms = 0.0;
};

struct ErrorStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t max_abs_index = 0U;
    size_t max_rel_index = 0U;
    float max_abs_lhs = 0.0f;
    float max_abs_rhs = 0.0f;
    float max_rel_lhs = 0.0f;
    float max_rel_rhs = 0.0f;
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

PerformanceCase MakeCase(int64_t batch, int64_t sample_count)
{
    PerformanceCase data;
    data.batch = batch;
    data.sample_count = sample_count;
    data.sample_interval = 1.0f;
    data.settle_band = 0.1f;
    data.pv.assign(static_cast<size_t>(batch * sample_count), 0.0f);
    data.sp.assign(static_cast<size_t>(batch * sample_count), 50.0f);
    data.lsl.assign(static_cast<size_t>(batch), 47.0f);
    data.usl.assign(static_cast<size_t>(batch), 53.0f);
    data.mv_variance.assign(static_cast<size_t>(batch), 0.08f);
    for (int64_t b = 0; b < batch; ++b) {
        const float drift = -0.25f + 0.5f * static_cast<float>(b) / static_cast<float>(std::max<int64_t>(1, batch - 1));
        for (int64_t i = 0; i < sample_count; ++i) {
            const float phase = static_cast<float>((i * 17 + b * 13) % 2048) / 2048.0f;
            const float target = 50.0f + 0.35f * std::sin(phase * 6.283185307179586f);
            const float oscillation = 0.6f * std::sin(phase * 12.566370614359172f + static_cast<float>(b) * 0.013f);
            const float ripple = 0.08f * std::cos(phase * 50.26548245743669f + static_cast<float>(b) * 0.031f);
            const size_t offset = static_cast<size_t>(b * sample_count + i);
            data.sp[offset] = target;
            data.pv[offset] = target + drift + oscillation + ripple;
        }
    }
    if (batch > 0 && sample_count > 1) {
        data.pv[0] = 53.8f;
        data.pv[1] = 46.2f;
        data.pv[data.pv.size() - 1U] = 54.0f;
    }
    return data;
}

void StoreMetrics(
    const PerformanceCase& data, int64_t b, double mean, double var_population, double var_sample, double out_count,
    double iae, double ise, double itae, double max_abs_error, double max_positive, double max_negative,
    double last_unsettled_time, std::vector<float>& metrics)
{
    constexpr double kEps = 1.0e-6;
    const double n = static_cast<double>(data.sample_count);
    const double lower = data.lsl[static_cast<size_t>(b)];
    const double upper = data.usl[static_cast<size_t>(b)];
    const double width = std::max(upper - lower, kEps);
    const double std_population = std::sqrt(std::max(0.0, var_population));
    const double std_sample = std::sqrt(std::max(0.0, var_sample));
    const double denom_sample = 3.0 * std::max(std_sample, kEps);
    const double denom_population = 3.0 * std::max(std_population, kEps);
    const double cpu = (upper - mean) / denom_sample;
    const double cpl = (mean - lower) / denom_sample;
    const double ppu = (upper - mean) / denom_population;
    const double ppl = (mean - lower) / denom_population;
    const double harris =
        std::min(1.0, std::max(0.0, static_cast<double>(data.mv_variance[static_cast<size_t>(b)]) /
                                      std::max(var_population, kEps)));
    const size_t last = static_cast<size_t>(b * data.sample_count + data.sample_count - 1);
    const double final_error = std::fabs(static_cast<double>(data.sp[last]) - data.pv[last]);
    const size_t out = static_cast<size_t>(b * PID_CONTROL_PERFORMANCE_METRIC_COUNT);
    metrics[out + 0U] = static_cast<float>(mean);
    metrics[out + 1U] = static_cast<float>(std_sample);
    metrics[out + 2U] = static_cast<float>(std_population);
    metrics[out + 3U] = static_cast<float>(width / (2.0 * denom_sample));
    metrics[out + 4U] = static_cast<float>(std::min(cpu, cpl));
    metrics[out + 5U] = static_cast<float>(width / (2.0 * denom_population));
    metrics[out + 6U] = static_cast<float>(std::min(ppu, ppl));
    metrics[out + 7U] = static_cast<float>(harris);
    metrics[out + 8U] = static_cast<float>(iae);
    metrics[out + 9U] = static_cast<float>(ise);
    metrics[out + 10U] = static_cast<float>(itae);
    metrics[out + 11U] = static_cast<float>(iae / n);
    metrics[out + 12U] = static_cast<float>(std::sqrt(ise / std::max(n * data.sample_interval, kEps)));
    metrics[out + 13U] = static_cast<float>(max_abs_error);
    metrics[out + 14U] = static_cast<float>(out_count / n);
    metrics[out + 15U] = static_cast<float>(out_count);
    metrics[out + 16U] = static_cast<float>(std::max(max_positive, 0.0) / width);
    metrics[out + 17U] = static_cast<float>(std::max(max_negative, 0.0) / width);
    metrics[out + 18U] = static_cast<float>(last_unsettled_time);
    metrics[out + 19U] = static_cast<float>(final_error);
}

void ComputeRange(const PerformanceCase& data, int64_t begin, int64_t end, std::vector<float>& metrics)
{
    for (int64_t b = begin; b < end; ++b) {
        double mean = 0.0;
        double m2 = 0.0;
        double count = 0.0;
        double out_count = 0.0;
        double iae = 0.0;
        double ise = 0.0;
        double itae = 0.0;
        double max_abs_error = 0.0;
        double max_positive = 0.0;
        double max_negative = 0.0;
        double last_unsettled_time = 0.0;
        for (int64_t i = 0; i < data.sample_count; ++i) {
            const size_t offset = static_cast<size_t>(b * data.sample_count + i);
            const double pv = data.pv[offset];
            const double sp = data.sp[offset];
            count += 1.0;
            const double delta = pv - mean;
            mean += delta / count;
            m2 += delta * (pv - mean);
            const double error = sp - pv;
            const double abs_error = std::fabs(error);
            const double time_value = static_cast<double>(i) * data.sample_interval;
            iae += abs_error * data.sample_interval;
            ise += error * error * data.sample_interval;
            itae += time_value * abs_error * data.sample_interval;
            max_abs_error = std::max(max_abs_error, abs_error);
            max_positive = std::max(max_positive, pv - sp);
            max_negative = std::max(max_negative, sp - pv);
            if (abs_error > data.settle_band) {
                last_unsettled_time = time_value + data.sample_interval;
            }
            if (pv < data.lsl[static_cast<size_t>(b)] || pv > data.usl[static_cast<size_t>(b)]) {
                out_count += 1.0;
            }
        }
        const double var_population = std::max(0.0, m2 / std::max(count, 1.0));
        const double var_sample = m2 / std::max(count - 1.0, 1.0);
        StoreMetrics(
            data, b, mean, var_population, var_sample, out_count, iae, ise, itae, max_abs_error, max_positive,
            max_negative, last_unsettled_time, metrics);
    }
}

std::vector<float> CpuMetrics(const PerformanceCase& data)
{
    std::vector<float> metrics(static_cast<size_t>(data.batch * PID_CONTROL_PERFORMANCE_METRIC_COUNT), 0.0f);
    ComputeRange(data, 0, data.batch, metrics);
    return metrics;
}

double MeasureCpuMs(const PerformanceCase& data, int threads, int iters, const char* guard_label)
{
    const int worker_count = static_cast<int>(std::min<int64_t>(std::max(1, threads), data.batch));
    std::vector<float> metrics(static_cast<size_t>(data.batch * PID_CONTROL_PERFORMANCE_METRIC_COUNT), 0.0f);
    volatile double guard = 0.0;
    if (worker_count <= 1) {
        auto start = Clock::now();
        for (int i = 0; i < iters; ++i) {
            ComputeRange(data, 0, data.batch, metrics);
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
                ComputeRange(data, begin, end, metrics);
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

NpuResult RunNpu(const PerformanceCase& data, int device_id, int iters, int warmup)
{
    NpuResult result;
    result.metrics.assign(static_cast<size_t>(data.batch * PID_CONTROL_PERFORMANCE_METRIC_COUNT), 0.0f);
    aclrtStream stream = nullptr;
    void* d_pv = nullptr;
    void* d_sp = nullptr;
    void* d_lsl = nullptr;
    void* d_usl = nullptr;
    void* d_mv_variance = nullptr;
    void* d_metrics = nullptr;
    void* workspace = nullptr;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    CHECK_ACL(aclrtCreateStream(&stream));
    d_pv = MallocDevice(data.pv.size() * sizeof(float));
    d_sp = MallocDevice(data.sp.size() * sizeof(float));
    d_lsl = MallocDevice(data.lsl.size() * sizeof(float));
    d_usl = MallocDevice(data.usl.size() * sizeof(float));
    d_mv_variance = MallocDevice(data.mv_variance.size() * sizeof(float));
    d_metrics = MallocDevice(result.metrics.size() * sizeof(float));
    const uint64_t workspace_size =
        aclnnPidControlPerformanceMetricsGetWorkspaceSize(data.batch, data.sample_count);
    workspace = MallocDevice(workspace_size);

    auto copy_inputs = [&]() {
        CHECK_ACL(aclrtMemcpy(d_pv, data.pv.size() * sizeof(float), data.pv.data(), data.pv.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_sp, data.sp.size() * sizeof(float), data.sp.data(), data.sp.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_lsl, data.lsl.size() * sizeof(float), data.lsl.data(), data.lsl.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_usl, data.usl.size() * sizeof(float), data.usl.data(), data.usl.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_mv_variance, data.mv_variance.size() * sizeof(float), data.mv_variance.data(),
                              data.mv_variance.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    };
    auto run_once = [&]() {
        const int32_t ret = aclnnPidControlPerformanceMetrics(
            d_pv, d_sp, d_lsl, d_usl, d_mv_variance, d_metrics, data.batch, data.sample_count, data.sample_interval,
            data.settle_band, workspace, workspace_size, stream);
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
    aclrtFree(d_mv_variance);
    aclrtFree(d_usl);
    aclrtFree(d_lsl);
    aclrtFree(d_sp);
    aclrtFree(d_pv);
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
        const float denom = std::max(std::fabs(lhs[i]), 1.0f);
        const float rel_err = abs_err / denom;
        if (abs_err > stats.max_abs) {
            stats.max_abs = abs_err;
            stats.max_abs_index = i;
            stats.max_abs_lhs = lhs[i];
            stats.max_abs_rhs = rhs[i];
        }
        if (rel_err > stats.max_rel) {
            stats.max_rel = rel_err;
            stats.max_rel_index = i;
            stats.max_rel_lhs = lhs[i];
            stats.max_rel_rhs = rhs[i];
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
        const PerformanceCase data = MakeCase(batch, sample_count);
        const std::vector<float> cpu = CpuMetrics(data);
        const double cpu_single_ms =
            MeasureCpuMs(data, 1, std::max(1, std::min(3, iters)), "cpu_single_guard");
        const double cpu_parallel_ms = MeasureCpuMs(data, cpu_threads, iters, "cpu_parallel_guard");
        const NpuResult npu = RunNpu(data, device_id, iters, warmup);
        const ErrorStats err = ComputeErrorStats(cpu, npu.metrics);

        std::cout << "op=PidControlPerformanceMetrics batch=" << batch << " sample_count=" << sample_count
                  << " iters=" << iters << " warmup=" << warmup << std::endl;
        std::cout << "cpu_threads=" << cpu_threads << std::endl;
        std::cout << "cpu_single_ms_avg=" << cpu_single_ms << std::endl;
        std::cout << "cpu_parallel_ms_avg=" << cpu_parallel_ms << std::endl;
        std::cout << "npu_kernel_ms_avg=" << npu.kernel_ms << std::endl;
        std::cout << "npu_e2e_ms_avg=" << npu.e2e_ms << std::endl;
        std::cout << "speedup_npu_kernel_vs_cpu_parallel=" << (cpu_parallel_ms / npu.kernel_ms) << std::endl;
        std::cout << "speedup_npu_e2e_vs_cpu_parallel=" << (cpu_parallel_ms / npu.e2e_ms) << std::endl;
        std::cout << "speedup_npu_kernel_vs_cpu_single=" << (cpu_single_ms / npu.kernel_ms) << std::endl;
        std::cout << "max_abs_err=" << err.max_abs << std::endl;
        std::cout << "max_abs_err_metric_index=" << (err.max_abs_index % PID_CONTROL_PERFORMANCE_METRIC_COUNT)
                  << std::endl;
        std::cout << "max_abs_err_loop_index=" << (err.max_abs_index / PID_CONTROL_PERFORMANCE_METRIC_COUNT)
                  << std::endl;
        std::cout << "max_abs_err_cpu=" << err.max_abs_lhs << std::endl;
        std::cout << "max_abs_err_npu=" << err.max_abs_rhs << std::endl;
        std::cout << "max_rel_err=" << err.max_rel << std::endl;
        std::cout << "max_rel_err_metric_index=" << (err.max_rel_index % PID_CONTROL_PERFORMANCE_METRIC_COUNT)
                  << std::endl;
        std::cout << "max_rel_err_loop_index=" << (err.max_rel_index / PID_CONTROL_PERFORMANCE_METRIC_COUNT)
                  << std::endl;
        std::cout << "max_rel_err_cpu=" << err.max_rel_lhs << std::endl;
        std::cout << "max_rel_err_npu=" << err.max_rel_rhs << std::endl;
        std::cout << "metric_order=mean_pv,std_pv_sample,std_pv_population,cp,cpk,pp,ppk,harris_index,iae,ise,"
                     "itae,mae,rmse,max_abs_error,out_ratio,out_count,overshoot,undershoot,settling_time,"
                     "final_abs_error"
                  << std::endl;
        return (err.max_abs < 1.0e-1f || err.max_rel < 1.0e-4f) ? 0 : 2;
    } catch (const std::exception& ex) {
        std::cerr << "benchmark failed: " << ex.what() << std::endl;
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }
}
