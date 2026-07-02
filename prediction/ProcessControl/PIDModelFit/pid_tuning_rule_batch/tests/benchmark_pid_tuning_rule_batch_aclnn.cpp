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
#include "pid_tuning_rule_batch_host.h"

namespace {

using Clock = std::chrono::steady_clock;

struct TuningCase {
    std::vector<float> gain;
    std::vector<float> tau;
    std::vector<float> theta;
    std::vector<float> lambda;
    int64_t batch = 0;
};

struct TuningResult {
    std::vector<float> pid;
    std::vector<float> diagnostics;
};

struct NpuResult {
    TuningResult values;
    double kernel_ms = 0.0;
    double e2e_ms = 0.0;
};

struct ErrorStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t max_index = 0U;
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

TuningCase MakeCase(int64_t batch)
{
    TuningCase data;
    data.batch = batch;
    data.gain.resize(static_cast<size_t>(batch));
    data.tau.resize(static_cast<size_t>(batch));
    data.theta.resize(static_cast<size_t>(batch));
    data.lambda.resize(static_cast<size_t>(batch));
    for (int64_t i = 0; i < batch; ++i) {
        const float idx = static_cast<float>(i);
        data.gain[static_cast<size_t>(i)] = 0.6f + 0.013f * static_cast<float>((i * 37) % 185);
        data.tau[static_cast<size_t>(i)] = 5.0f + 0.25f * static_cast<float>((i * 19) % 300);
        const float ratio = 0.04f + 0.002f * static_cast<float>((i * 23) % 380);
        data.theta[static_cast<size_t>(i)] = data.tau[static_cast<size_t>(i)] * ratio;
        data.lambda[static_cast<size_t>(i)] =
            data.theta[static_cast<size_t>(i)] * (1.0f + 0.01f * static_cast<float>((i * 29) % 500));
        if ((i % 9973) == 0) {
            data.gain[static_cast<size_t>(i)] = (i == 0) ? 0.0f : data.gain[static_cast<size_t>(i)];
        }
        (void)idx;
    }
    return data;
}

void ComputeOne(const TuningCase& data, int64_t i, TuningResult& out)
{
    constexpr float kEps = 1.0e-6f;
    const size_t idx = static_cast<size_t>(i);
    const float gain = data.gain[idx];
    const float tau_raw = data.tau[idx];
    const float theta_raw = data.theta[idx];
    const float lambda_raw = data.lambda[idx];
    const bool valid = AbsF(gain) > kEps && tau_raw > kEps && theta_raw > kEps && lambda_raw > kEps;
    const float safe_gain = AbsF(gain) > kEps ? gain : 1.0f;
    const float tau = std::max(tau_raw, kEps);
    const float theta = std::max(theta_raw, kEps);
    const float lambda = std::max(lambda_raw, kEps);
    const float ratio = theta / tau;
    const float lambda_ratio = lambda / theta;

    auto store = [&](int rule, float kp, float ti, float td) {
        const size_t param_base =
            static_cast<size_t>((i * PID_TUNING_RULE_BATCH_RULE_COUNT + rule) * PID_TUNING_RULE_BATCH_PARAM_COUNT);
        const size_t diag_base = static_cast<size_t>(
            (i * PID_TUNING_RULE_BATCH_RULE_COUNT + rule) * PID_TUNING_RULE_BATCH_DIAGNOSTIC_COUNT);
        const float ki = kp / std::max(ti, kEps);
        const float kd = kp * td;
        out.pid[param_base + 0U] = valid ? kp : 0.0f;
        out.pid[param_base + 1U] = valid ? ki : 0.0f;
        out.pid[param_base + 2U] = valid ? kd : 0.0f;
        out.diagnostics[diag_base + 0U] = valid ? 1.0f : 0.0f;
        out.diagnostics[diag_base + 1U] = ratio;
        out.diagnostics[diag_base + 2U] = valid ? AbsF(kp) + tau * AbsF(ki) + AbsF(kd) / tau : 0.0f;
        out.diagnostics[diag_base + 3U] = lambda_ratio;
    };

    float kp = 1.2f * tau / (safe_gain * theta);
    store(0, kp, 2.0f * theta, 0.5f * theta);

    kp = (tau + 0.5f * theta) / (safe_gain * (lambda + 0.5f * theta));
    store(1, kp, tau + 0.5f * theta, tau * theta / std::max(2.0f * tau + theta, kEps));

    kp = (tau / (safe_gain * theta)) * (1.3333333333333333f + theta / (4.0f * tau));
    store(
        2, kp, theta * (32.0f + 6.0f * ratio) / std::max(13.0f + 8.0f * ratio, kEps),
        theta * 4.0f / std::max(11.0f + 2.0f * ratio, kEps));
}

void ComputeRange(const TuningCase& data, int64_t begin, int64_t end, TuningResult& out)
{
    for (int64_t i = begin; i < end; ++i) {
        ComputeOne(data, i, out);
    }
}

TuningResult CpuTuning(const TuningCase& data)
{
    TuningResult out;
    out.pid.assign(
        static_cast<size_t>(data.batch * PID_TUNING_RULE_BATCH_RULE_COUNT * PID_TUNING_RULE_BATCH_PARAM_COUNT), 0.0f);
    out.diagnostics.assign(
        static_cast<size_t>(data.batch * PID_TUNING_RULE_BATCH_RULE_COUNT * PID_TUNING_RULE_BATCH_DIAGNOSTIC_COUNT),
        0.0f);
    ComputeRange(data, 0, data.batch, out);
    return out;
}

double MeasureCpuMs(const TuningCase& data, int threads, int iters, const char* guard_label)
{
    const int worker_count = static_cast<int>(std::min<int64_t>(std::max(1, threads), data.batch));
    TuningResult out;
    out.pid.assign(
        static_cast<size_t>(data.batch * PID_TUNING_RULE_BATCH_RULE_COUNT * PID_TUNING_RULE_BATCH_PARAM_COUNT), 0.0f);
    out.diagnostics.assign(
        static_cast<size_t>(data.batch * PID_TUNING_RULE_BATCH_RULE_COUNT * PID_TUNING_RULE_BATCH_DIAGNOSTIC_COUNT),
        0.0f);
    volatile double guard = 0.0;
    if (worker_count <= 1) {
        const auto start = Clock::now();
        for (int iter = 0; iter < iters; ++iter) {
            ComputeRange(data, 0, data.batch, out);
            guard += out.pid[static_cast<size_t>(iter) % out.pid.size()];
        }
        const auto end = Clock::now();
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
                    done_cv.notify_one();
                }
            }
        });
    }

    const auto start = Clock::now();
    for (int iter = 0; iter < iters; ++iter) {
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
        guard += out.pid[static_cast<size_t>(iter) % out.pid.size()];
    }
    const auto end = Clock::now();
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

NpuResult RunNpu(const TuningCase& data, int iters)
{
    NpuResult out;
    out.values.pid.assign(
        static_cast<size_t>(data.batch * PID_TUNING_RULE_BATCH_RULE_COUNT * PID_TUNING_RULE_BATCH_PARAM_COUNT), 0.0f);
    out.values.diagnostics.assign(
        static_cast<size_t>(data.batch * PID_TUNING_RULE_BATCH_RULE_COUNT * PID_TUNING_RULE_BATCH_DIAGNOSTIC_COUNT),
        0.0f);

    const size_t input_bytes = data.gain.size() * sizeof(float);
    const size_t pid_bytes = out.values.pid.size() * sizeof(float);
    const size_t diag_bytes = out.values.diagnostics.size() * sizeof(float);
    const uint64_t workspace_size = aclnnPidTuningRuleBatchGetWorkspaceSize(data.batch);
    void* d_gain = nullptr;
    void* d_tau = nullptr;
    void* d_theta = nullptr;
    void* d_lambda = nullptr;
    void* d_pid = nullptr;
    void* d_diag = nullptr;
    void* workspace = nullptr;
    aclrtStream stream = nullptr;
    aclrtEvent start_event = nullptr;
    aclrtEvent end_event = nullptr;

    CHECK_ACL(aclrtCreateStream(&stream));
    CHECK_ACL(aclrtCreateEvent(&start_event));
    CHECK_ACL(aclrtCreateEvent(&end_event));
    CHECK_ACL(aclrtMalloc(&d_gain, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_tau, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_theta, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_lambda, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_pid, pid_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_diag, diag_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_gain, input_bytes, data.gain.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_tau, input_bytes, data.tau.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_theta, input_bytes, data.theta.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_lambda, input_bytes, data.lambda.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    auto launch = [&]() {
        const int32_t ret = aclnnPidTuningRuleBatch(
            d_gain, d_tau, d_theta, d_lambda, d_pid, d_diag, data.batch, workspace, workspace_size, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclnnPidTuningRuleBatch failed, ret=" + std::to_string(ret));
        }
    };

    launch();
    CHECK_ACL(aclrtSynchronizeStream(stream));
    double kernel_sum = 0.0;
    for (int iter = 0; iter < iters; ++iter) {
        CHECK_ACL(aclrtRecordEvent(start_event, stream));
        launch();
        CHECK_ACL(aclrtRecordEvent(end_event, stream));
        CHECK_ACL(aclrtSynchronizeEvent(end_event));
        float elapsed = 0.0f;
        CHECK_ACL(aclrtEventElapsedTime(&elapsed, start_event, end_event));
        kernel_sum += static_cast<double>(elapsed);
    }
    out.kernel_ms = kernel_sum / static_cast<double>(iters);

    const auto e2e_start = Clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        CHECK_ACL(aclrtMemcpyAsync(d_gain, input_bytes, data.gain.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream));
        CHECK_ACL(aclrtMemcpyAsync(d_tau, input_bytes, data.tau.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream));
        CHECK_ACL(
            aclrtMemcpyAsync(d_theta, input_bytes, data.theta.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream));
        CHECK_ACL(aclrtMemcpyAsync(
            d_lambda, input_bytes, data.lambda.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream));
        launch();
        CHECK_ACL(aclrtMemcpyAsync(out.values.pid.data(), pid_bytes, d_pid, pid_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK_ACL(aclrtMemcpyAsync(
            out.values.diagnostics.data(), diag_bytes, d_diag, diag_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }
    const auto e2e_end = Clock::now();
    out.e2e_ms = MsSince(e2e_start, e2e_end) / static_cast<double>(iters);

    CHECK_ACL(aclrtDestroyEvent(end_event));
    CHECK_ACL(aclrtDestroyEvent(start_event));
    CHECK_ACL(aclrtFree(workspace));
    CHECK_ACL(aclrtFree(d_diag));
    CHECK_ACL(aclrtFree(d_pid));
    CHECK_ACL(aclrtFree(d_lambda));
    CHECK_ACL(aclrtFree(d_theta));
    CHECK_ACL(aclrtFree(d_tau));
    CHECK_ACL(aclrtFree(d_gain));
    CHECK_ACL(aclrtDestroyStream(stream));
    return out;
}

ErrorStats Compare(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    ErrorStats stats;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float abs_err = std::fabs(lhs[i] - rhs[i]);
        const float rel_err = abs_err / std::max(std::fabs(lhs[i]), 1.0e-6f);
        if (abs_err > stats.max_abs) {
            stats.max_abs = abs_err;
            stats.max_rel = rel_err;
            stats.max_index = i;
        }
    }
    return stats;
}

void RunCase(int64_t batch)
{
    const int cpu_iters = batch <= 4096 ? 200 : 50;
    const int npu_iters = batch <= 4096 ? 200 : 50;
    const TuningCase data = MakeCase(batch);
    const TuningResult cpu_ref = CpuTuning(data);
    const double cpu_single_ms = MeasureCpuMs(data, 1, cpu_iters, "cpu_single_guard");
    const double cpu_64_ms = MeasureCpuMs(data, 64, cpu_iters, "cpu_64_guard");
    const NpuResult npu = RunNpu(data, npu_iters);
    const ErrorStats pid_err = Compare(cpu_ref.pid, npu.values.pid);
    const ErrorStats diag_err = Compare(cpu_ref.diagnostics, npu.values.diagnostics);

    std::cout << "CASE B=" << batch << std::endl;
    std::cout << "cpu_single_ms=" << cpu_single_ms << std::endl;
    std::cout << "cpu_64_ms=" << cpu_64_ms << std::endl;
    std::cout << "npu_kernel_ms=" << npu.kernel_ms << std::endl;
    std::cout << "npu_e2e_ms=" << npu.e2e_ms << std::endl;
    std::cout << "speedup_e2e_vs_cpu64=" << (cpu_64_ms / npu.e2e_ms) << std::endl;
    std::cout << "pid_max_abs=" << pid_err.max_abs << " pid_max_rel=" << pid_err.max_rel
              << " pid_idx=" << pid_err.max_index << std::endl;
    std::cout << "diag_max_abs=" << diag_err.max_abs << " diag_max_rel=" << diag_err.max_rel
              << " diag_idx=" << diag_err.max_index << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    const int32_t device_id = argc > 1 ? static_cast<int32_t>(std::strtol(argv[1], nullptr, 10)) : 0;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    if (argc > 2) {
        RunCase(std::strtoll(argv[2], nullptr, 10));
    } else {
        RunCase(4096);
        RunCase(65536);
        RunCase(262144);
        RunCase(1048576);
    }
    CHECK_ACL(aclrtResetDevice(device_id));
    CHECK_ACL(aclFinalize());
    return 0;
}
