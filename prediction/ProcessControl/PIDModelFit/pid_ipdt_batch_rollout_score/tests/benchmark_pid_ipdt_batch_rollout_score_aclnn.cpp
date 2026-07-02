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
#include "pid_ipdt_batch_rollout_score_host.h"

namespace {

using Clock = std::chrono::steady_clock;

struct ClosedLoopCase {
    std::vector<float> b;
    std::vector<int32_t> delay;
    std::vector<float> y0;
    std::vector<float> sp;
    std::vector<float> kp;
    std::vector<float> ki;
    std::vector<float> kd;
    int64_t batch = 0;
    int64_t candidates = 0;
    int64_t sim_steps = 0;
    int64_t candidate_tile = 64;
    float sample_interval = 1.0f;
    float settle_band = 0.02f;
    float overshoot_weight = 50.0f;
    float settling_weight = 0.02f;
    float control_weight = 0.001f;
};

struct NpuResult {
    std::vector<float> best_result;
    std::vector<int32_t> best_idx;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
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

ClosedLoopCase MakeCase(int64_t batch, int64_t candidates, int64_t sim_steps)
{
    ClosedLoopCase data;
    data.batch = batch;
    data.candidates = candidates;
    data.sim_steps = sim_steps;
    data.b.assign(static_cast<size_t>(batch), 0.0f);
    data.delay.assign(static_cast<size_t>(batch), 0);
    data.y0.assign(static_cast<size_t>(batch), 0.0f);
    data.sp.assign(static_cast<size_t>(batch), 1.0f);
    data.kp.assign(static_cast<size_t>(candidates), 0.0f);
    data.ki.assign(static_cast<size_t>(candidates), 0.0f);
    data.kd.assign(static_cast<size_t>(candidates), 0.0f);

    for (int64_t i = 0; i < batch; ++i) {
        const float tau = 8.0f + 0.2f * static_cast<float>(i % 160);
        const float gain = 0.6f + 0.01f * static_cast<float>(i % 120);
        const float a = std::exp(-data.sample_interval / tau);
        data.b[static_cast<size_t>(i)] = gain * (1.0f - a);
        data.delay[static_cast<size_t>(i)] = static_cast<int32_t>(1 + (i % 16));
        data.y0[static_cast<size_t>(i)] = -0.1f + 0.002f * static_cast<float>(i % 100);
    }
    for (int64_t c = 0; c < candidates; ++c) {
        const float ratio = static_cast<float>(c) / static_cast<float>(std::max<int64_t>(1, candidates - 1));
        data.kp[static_cast<size_t>(c)] = 0.05f + 2.95f * ratio;
        data.ki[static_cast<size_t>(c)] = 0.35f * ratio;
        data.kd[static_cast<size_t>(c)] = 0.25f * ratio;
    }
    return data;
}

void StoreBest(
    const ClosedLoopCase& data, int64_t loop, float score, float kp, float ki, float kd, float iae, float ise,
    float overshoot, float settling, int32_t idx, std::vector<float>& best_result, std::vector<int32_t>& best_idx)
{
    const size_t out = static_cast<size_t>(loop * PID_IPDT_BATCH_ROLLOUT_RESULT_COUNT);
    best_result[out + 0U] = score;
    best_result[out + 1U] = kp;
    best_result[out + 2U] = ki;
    best_result[out + 3U] = kd;
    best_result[out + 4U] = iae;
    best_result[out + 5U] = ise;
    best_result[out + 6U] = overshoot;
    best_result[out + 7U] = settling;
    best_idx[static_cast<size_t>(loop)] = idx;
    (void)data;
}

void ComputeRange(
    const ClosedLoopCase& data, int64_t begin, int64_t end, std::vector<float>& best_result,
    std::vector<int32_t>& best_idx)
{
    for (int64_t loop = begin; loop < end; ++loop) {
        const float b = data.b[static_cast<size_t>(loop)];
        const int delay_len = std::max(1, std::min(32, data.delay[static_cast<size_t>(loop)] + 1));
        const float initial_y = data.y0[static_cast<size_t>(loop)];
        const float target = data.sp[static_cast<size_t>(loop)];
        float best_score = 3.402823466e30f;
        float out_kp = 0.0f;
        float out_ki = 0.0f;
        float out_kd = 0.0f;
        float out_iae = 0.0f;
        float out_ise = 0.0f;
        float out_overshoot = 0.0f;
        float out_settling = 0.0f;
        int32_t out_idx = 0;
        for (int64_t candidate = 0; candidate < data.candidates; ++candidate) {
            const float kp = data.kp[static_cast<size_t>(candidate)];
            const float ki = data.ki[static_cast<size_t>(candidate)];
            const float kd = data.kd[static_cast<size_t>(candidate)];
            float u_hist[32] = {0.0f};
            float y = initial_y;
            float integral = 0.0f;
            float prev_error = target - y;
            float iae = 0.0f;
            float ise = 0.0f;
            float max_overshoot = 0.0f;
            float settling_time = 0.0f;
            float control_energy = 0.0f;
            float time_value = 0.0f;
            for (int64_t step = 0; step < data.sim_steps; ++step) {
                const float error = target - y;
                integral += error * data.sample_interval;
                const float derivative = (error - prev_error) / data.sample_interval;
                float control = kp * error + ki * integral + kd * derivative;
                control = std::min(std::max(control, -10.0f), 10.0f);
                const float delayed_control = u_hist[0];
                for (int h = 0; h + 1 < delay_len; ++h) {
                    u_hist[h] = u_hist[h + 1];
                }
                u_hist[delay_len - 1] = control;
                y = y + b * delayed_control;  // IPDT integrator, a = 1
                const float response_error = target - y;
                const float abs_error = std::fabs(response_error);
                iae += abs_error * data.sample_interval;
                ise += response_error * response_error * data.sample_interval;
                max_overshoot = std::max(max_overshoot, y - target);
                control_energy += control * control * data.sample_interval;
                if (abs_error > data.settle_band) {
                    settling_time = time_value + data.sample_interval;
                }
                prev_error = error;
                time_value += data.sample_interval;
            }
            const float overshoot = std::max(max_overshoot, 0.0f);
            const float score = iae + data.overshoot_weight * overshoot + data.settling_weight * settling_time +
                                data.control_weight * control_energy;
            if (score < best_score) {
                best_score = score;
                out_kp = kp;
                out_ki = ki;
                out_kd = kd;
                out_iae = iae;
                out_ise = ise;
                out_overshoot = overshoot;
                out_settling = settling_time;
                out_idx = static_cast<int32_t>(candidate);
            }
        }
        StoreBest(data, loop, best_score, out_kp, out_ki, out_kd, out_iae, out_ise, out_overshoot, out_settling,
                  out_idx, best_result, best_idx);
    }
}

void CpuScore(
    const ClosedLoopCase& data, int threads, std::vector<float>& best_result, std::vector<int32_t>& best_idx)
{
    const int worker_count = static_cast<int>(std::min<int64_t>(std::max(1, threads), data.batch));
    best_result.assign(static_cast<size_t>(data.batch * PID_IPDT_BATCH_ROLLOUT_RESULT_COUNT), 0.0f);
    best_idx.assign(static_cast<size_t>(data.batch), 0);
    if (worker_count <= 1) {
        ComputeRange(data, 0, data.batch, best_result, best_idx);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int worker = 0; worker < worker_count; ++worker) {
        const int64_t begin = data.batch * worker / worker_count;
        const int64_t end = data.batch * (worker + 1) / worker_count;
        workers.emplace_back([&, begin, end]() { ComputeRange(data, begin, end, best_result, best_idx); });
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

double MeasureCpuMs(const ClosedLoopCase& data, int threads, int iters, const char* guard_label)
{
    const int worker_count = static_cast<int>(std::min<int64_t>(std::max(1, threads), data.batch));
    std::vector<float> best_result(static_cast<size_t>(data.batch * PID_IPDT_BATCH_ROLLOUT_RESULT_COUNT), 0.0f);
    std::vector<int32_t> best_idx(static_cast<size_t>(data.batch), 0);
    volatile double guard = 0.0;
    if (worker_count <= 1) {
        auto start = Clock::now();
        for (int i = 0; i < iters; ++i) {
            ComputeRange(data, 0, data.batch, best_result, best_idx);
            guard += best_result[static_cast<size_t>(i) % best_result.size()];
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
                ComputeRange(data, begin, end, best_result, best_idx);
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
        guard += best_result[static_cast<size_t>(i) % best_result.size()];
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

NpuResult RunNpu(const ClosedLoopCase& data, int device_id, int iters, int warmup)
{
    NpuResult result;
    result.best_result.assign(static_cast<size_t>(data.batch * PID_IPDT_BATCH_ROLLOUT_RESULT_COUNT), 0.0f);
    result.best_idx.assign(static_cast<size_t>(data.batch), 0);
    aclrtStream stream = nullptr;
    void* d_b = nullptr;
    void* d_delay = nullptr;
    void* d_y0 = nullptr;
    void* d_sp = nullptr;
    void* d_kp = nullptr;
    void* d_ki = nullptr;
    void* d_kd = nullptr;
    void* d_best_result = nullptr;
    void* d_best_idx = nullptr;
    void* workspace = nullptr;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    CHECK_ACL(aclrtCreateStream(&stream));
    d_b = MallocDevice(data.b.size() * sizeof(float));
    d_delay = MallocDevice(data.delay.size() * sizeof(int32_t));
    d_y0 = MallocDevice(data.y0.size() * sizeof(float));
    d_sp = MallocDevice(data.sp.size() * sizeof(float));
    d_kp = MallocDevice(data.kp.size() * sizeof(float));
    d_ki = MallocDevice(data.ki.size() * sizeof(float));
    d_kd = MallocDevice(data.kd.size() * sizeof(float));
    d_best_result = MallocDevice(result.best_result.size() * sizeof(float));
    d_best_idx = MallocDevice(result.best_idx.size() * sizeof(int32_t));
    const uint64_t workspace_size =
        aclnnPidIpdtBatchRolloutScoreGetWorkspaceSize(data.batch, data.candidates, data.sim_steps, data.candidate_tile);
    workspace = MallocDevice(workspace_size);

    auto copy_inputs = [&]() {
        CHECK_ACL(aclrtMemcpy(d_b, data.b.size() * sizeof(float), data.b.data(), data.b.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_delay, data.delay.size() * sizeof(int32_t), data.delay.data(),
                              data.delay.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_y0, data.y0.size() * sizeof(float), data.y0.data(), data.y0.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_sp, data.sp.size() * sizeof(float), data.sp.data(), data.sp.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_kp, data.kp.size() * sizeof(float), data.kp.data(), data.kp.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_ki, data.ki.size() * sizeof(float), data.ki.data(), data.ki.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(d_kd, data.kd.size() * sizeof(float), data.kd.data(), data.kd.size() * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));
    };
    auto launch_once = [&]() {
        const int32_t ret = aclnnPidIpdtBatchRolloutScore(
            d_b, d_delay, d_y0, d_sp, d_kp, d_ki, d_kd, d_best_result, d_best_idx, data.batch, data.candidates,
            data.sim_steps, data.candidate_tile, data.sample_interval, data.settle_band, data.overshoot_weight,
            data.settling_weight, data.control_weight, workspace, workspace_size, stream);
        CHECK_ACL(static_cast<aclError>(ret));
    };
    copy_inputs();
    for (int i = 0; i < warmup; ++i) {
        launch_once();
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

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
        CHECK_ACL(aclrtMemcpy(result.best_result.data(), result.best_result.size() * sizeof(float), d_best_result,
                              result.best_result.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(result.best_idx.data(), result.best_idx.size() * sizeof(int32_t), d_best_idx,
                              result.best_idx.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
        section_end = Clock::now();
        d2h_total += MsSince(section_start, section_end);
        e2e_total += MsSince(e2e_start, section_end);
    }
    result.h2d_ms = h2d_total / static_cast<double>(iters);
    result.kernel_ms = kernel_total / static_cast<double>(iters);
    result.d2h_ms = d2h_total / static_cast<double>(iters);
    result.e2e_ms = e2e_total / static_cast<double>(iters);

    CHECK_ACL(aclrtMemcpy(result.best_result.data(), result.best_result.size() * sizeof(float), d_best_result,
                          result.best_result.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(result.best_idx.data(), result.best_idx.size() * sizeof(int32_t), d_best_idx,
                          result.best_idx.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));

    aclrtFree(workspace);
    aclrtFree(d_best_idx);
    aclrtFree(d_best_result);
    aclrtFree(d_kd);
    aclrtFree(d_ki);
    aclrtFree(d_kp);
    aclrtFree(d_sp);
    aclrtFree(d_y0);
    aclrtFree(d_delay);
    aclrtFree(d_b);
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

ErrorStats ComputeQualityErrorStats(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    ErrorStats stats;
    constexpr int kQualityIndices[] = {0, 4, 5, 6, 7};
    const size_t loop_count = lhs.size() / PID_IPDT_BATCH_ROLLOUT_RESULT_COUNT;
    for (size_t loop = 0; loop < loop_count; ++loop) {
        const size_t base = loop * PID_IPDT_BATCH_ROLLOUT_RESULT_COUNT;
        for (int index : kQualityIndices) {
            const size_t offset = base + static_cast<size_t>(index);
            const float abs_err = std::fabs(lhs[offset] - rhs[offset]);
            const float rel_err = abs_err / std::max(std::fabs(lhs[offset]), 1.0f);
            if (abs_err > stats.max_abs) {
                stats.max_abs = abs_err;
                stats.max_abs_index = offset;
                stats.max_abs_lhs = lhs[offset];
                stats.max_abs_rhs = rhs[offset];
            }
            if (rel_err > stats.max_rel) {
                stats.max_rel = rel_err;
                stats.max_rel_index = offset;
                stats.max_rel_lhs = lhs[offset];
                stats.max_rel_rhs = rhs[offset];
            }
        }
    }
    return stats;
}

int CountDifferentIdx(const std::vector<int32_t>& lhs, const std::vector<int32_t>& rhs)
{
    int count = 0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            ++count;
        }
    }
    return count;
}

int CountZeroScores(const std::vector<float>& values)
{
    int count = 0;
    const size_t loop_count = values.size() / PID_IPDT_BATCH_ROLLOUT_RESULT_COUNT;
    for (size_t loop = 0; loop < loop_count; ++loop) {
        if (values[loop * PID_IPDT_BATCH_ROLLOUT_RESULT_COUNT] == 0.0f) {
            ++count;
        }
    }
    return count;
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
                  << " <device_id> [batch=128] [candidates=1024] [sim_steps=1024] [candidate_tile=0:auto] [iters=5]"
                     " [warmup=2] [cpu_threads=hardware]"
                  << std::endl;
        return 1;
    }
    const int device_id = std::atoi(argv[1]);
    const int64_t batch = argc > 2 ? std::atoll(argv[2]) : 128;
    const int64_t candidates = argc > 3 ? std::atoll(argv[3]) : 1024;
    const int64_t sim_steps = argc > 4 ? std::atoll(argv[4]) : 1024;
    const int64_t candidate_tile = argc > 5 ? std::atoll(argv[5]) : 0;  // 0 => auto: host picks min(C, kLane=768)
    const int iters = argc > 6 ? std::atoi(argv[6]) : 5;
    const int warmup = argc > 7 ? std::atoi(argv[7]) : 2;
    const int cpu_threads = ResolveThreadCount(argc > 8 ? std::atoi(argv[8]) : 0);
    if (batch <= 0 || candidates <= 0 || sim_steps <= 0 || iters <= 0 || warmup < 0) {
        std::cerr << "batch/candidates/sim_steps/candidate_tile/iters must be positive; warmup must be non-negative"
                  << std::endl;
        return 1;
    }

    try {
        ClosedLoopCase data = MakeCase(batch, candidates, sim_steps);
        data.candidate_tile = candidate_tile;
        std::vector<float> cpu_result;
        std::vector<int32_t> cpu_idx;
        CpuScore(data, cpu_threads, cpu_result, cpu_idx);
        const double cpu_single_ms = MeasureCpuMs(data, 1, std::max(1, std::min(2, iters)), "cpu_single_guard");
        const double cpu_parallel_ms = MeasureCpuMs(data, cpu_threads, iters, "cpu_parallel_guard");
        const NpuResult npu = RunNpu(data, device_id, iters, warmup);
        const ErrorStats err = ComputeErrorStats(cpu_result, npu.best_result);
        const ErrorStats quality_err = ComputeQualityErrorStats(cpu_result, npu.best_result);
        const int idx_diff = CountDifferentIdx(cpu_idx, npu.best_idx);
        const int npu_zero_scores = CountZeroScores(npu.best_result);

        std::cout << "op=PidIpdtBatchRolloutScore batch=" << batch << " candidates=" << candidates
                  << " sim_steps=" << sim_steps << " candidate_tile=" << candidate_tile << " iters=" << iters
                  << " warmup=" << warmup << std::endl;
        std::cout << "simulated_steps=" << (static_cast<double>(batch) * candidates * sim_steps) << std::endl;
        std::cout << "cpu_threads=" << cpu_threads << std::endl;
        std::cout << "cpu_single_ms_avg=" << cpu_single_ms << std::endl;
        std::cout << "cpu_parallel_ms_avg=" << cpu_parallel_ms << std::endl;
        std::cout << "npu_h2d_ms_avg=" << npu.h2d_ms << std::endl;
        std::cout << "npu_kernel_ms_avg=" << npu.kernel_ms << std::endl;
        std::cout << "npu_d2h_ms_avg=" << npu.d2h_ms << std::endl;
        std::cout << "npu_e2e_ms_avg=" << npu.e2e_ms << std::endl;
        std::cout << "speedup_npu_kernel_vs_cpu_parallel=" << (cpu_parallel_ms / npu.kernel_ms) << std::endl;
        std::cout << "speedup_npu_e2e_vs_cpu_parallel=" << (cpu_parallel_ms / npu.e2e_ms) << std::endl;
        std::cout << "speedup_npu_kernel_vs_cpu_single=" << (cpu_single_ms / npu.kernel_ms) << std::endl;
        std::cout << "max_abs_err=" << err.max_abs << std::endl;
        std::cout << "max_rel_err=" << err.max_rel << std::endl;
        std::cout << "max_quality_abs_err=" << quality_err.max_abs << std::endl;
        std::cout << "max_quality_rel_err=" << quality_err.max_rel << std::endl;
        std::cout << "best_idx_diff_count=" << idx_diff << std::endl;
        std::cout << "npu_zero_score_count=" << npu_zero_scores << std::endl;
        std::cout << "result_order=best_score,best_kp,best_ki,best_kd,best_iae,best_ise,best_overshoot,"
                     "best_settling_time"
                  << std::endl;
        return (quality_err.max_rel < 1.0e-3f && npu_zero_scores == 0) ? 0 : 2;
    } catch (const std::exception& ex) {
        std::cerr << "benchmark failed: " << ex.what() << std::endl;
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }
}
