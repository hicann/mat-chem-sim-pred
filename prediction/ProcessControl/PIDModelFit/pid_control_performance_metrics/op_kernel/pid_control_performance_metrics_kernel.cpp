/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct ControlPerformanceTilingData {
    uint32_t batch;
    uint32_t sample_count;
    uint32_t core_num;
    uint32_t metric_count;
    float sample_interval;
    float settle_band;
};

constexpr uint32_t kMetricCount = 20U;
constexpr float kEpsilon = 1.0e-6f;

__aicore__ inline float SqrtNewton(float value)
{
    if (value <= 0.0f) {
        return 0.0f;
    }
    float x = value > 1.0f ? value : 1.0f;
    for (int i = 0; i < 8; ++i) {
        x = 0.5f * (x + value / x);
    }
    return x;
}

__aicore__ inline float MaxF(float lhs, float rhs)
{
    return lhs > rhs ? lhs : rhs;
}

__aicore__ inline float MinF(float lhs, float rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline float AbsF(float value)
{
    return value >= 0.0f ? value : -value;
}

}  // namespace

extern "C" __global__ __aicore__ void pid_control_performance_metrics_kernel(
    GM_ADDR pv, GM_ADDR sp, GM_ADDR lsl, GM_ADDR usl, GM_ADDR mv_variance, GM_ADDR metrics, GM_ADDR tiling)
{
    const __gm__ ControlPerformanceTilingData* tiling_data =
        reinterpret_cast<const __gm__ ControlPerformanceTilingData*>(tiling);
    const uint32_t batch = tiling_data->batch;
    const uint32_t sample_count = tiling_data->sample_count;
    const uint32_t core_num = tiling_data->core_num;
    const float sample_interval = tiling_data->sample_interval;
    const float settle_band = tiling_data->settle_band;
    const uint32_t core_idx = GetBlockIdx();

    if (batch == 0U || sample_count == 0U || core_num == 0U) {
        return;
    }

    GlobalTensor<float> pv_gm;
    GlobalTensor<float> sp_gm;
    GlobalTensor<float> lsl_gm;
    GlobalTensor<float> usl_gm;
    GlobalTensor<float> mv_variance_gm;
    GlobalTensor<float> metrics_gm;
    pv_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pv));
    sp_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sp));
    lsl_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lsl));
    usl_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(usl));
    mv_variance_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mv_variance));
    metrics_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(metrics));

    const uint32_t loops_per_core = (batch + core_num - 1U) / core_num;
    const uint32_t start_loop = core_idx * loops_per_core;
    uint32_t end_loop = start_loop + loops_per_core;
    if (end_loop > batch) {
        end_loop = batch;
    }

    for (uint32_t loop = start_loop; loop < end_loop; ++loop) {
        const float lower = lsl_gm.GetValue(loop);
        const float upper = usl_gm.GetValue(loop);
        const float spec_width = MaxF(upper - lower, kEpsilon);
        const uint64_t base = static_cast<uint64_t>(loop) * sample_count;

        float mean = 0.0f;
        float m2 = 0.0f;
        float n = 0.0f;
        float out_count = 0.0f;
        float iae = 0.0f;
        float ise = 0.0f;
        float itae = 0.0f;
        float max_abs_error = 0.0f;
        float max_positive = 0.0f;
        float max_negative = 0.0f;
        float last_unsettled_time = 0.0f;
        float time_value = 0.0f;

        for (uint32_t i = 0; i < sample_count; ++i) {
            const uint64_t offset = base + i;
            const float pv_value = pv_gm.GetValue(offset);
            const float sp_value = sp_gm.GetValue(offset);
            n += 1.0f;
            const float delta = pv_value - mean;
            mean += delta / n;
            const float delta2 = pv_value - mean;
            m2 += delta * delta2;

            const float error = sp_value - pv_value;
            const float abs_error = AbsF(error);
            iae += abs_error * sample_interval;
            ise += error * error * sample_interval;
            itae += time_value * abs_error * sample_interval;
            max_abs_error = MaxF(max_abs_error, abs_error);
            max_positive = MaxF(max_positive, pv_value - sp_value);
            max_negative = MaxF(max_negative, sp_value - pv_value);
            if (abs_error > settle_band) {
                last_unsettled_time = time_value + sample_interval;
            }
            if (pv_value < lower || pv_value > upper) {
                out_count += 1.0f;
            }
            time_value += sample_interval;
        }

        const float var_population = MaxF(0.0f, m2 / n);
        const float var_sample = sample_count > 1U ? m2 / (n - 1.0f) : var_population;
        const float std_population = SqrtNewton(var_population);
        const float std_sample = SqrtNewton(var_sample);
        const float denom_sample = 3.0f * MaxF(std_sample, kEpsilon);
        const float denom_population = 3.0f * MaxF(std_population, kEpsilon);
        const float cpu = (upper - mean) / denom_sample;
        const float cpl = (mean - lower) / denom_sample;
        const float ppu = (upper - mean) / denom_population;
        const float ppl = (mean - lower) / denom_population;
        const float harris = MinF(1.0f, MaxF(0.0f, mv_variance_gm.GetValue(loop) / MaxF(var_population, kEpsilon)));
        const float final_error = AbsF(sp_gm.GetValue(base + sample_count - 1U) - pv_gm.GetValue(base + sample_count - 1U));
        const uint64_t out_base = static_cast<uint64_t>(loop) * kMetricCount;

        metrics_gm.SetValue(out_base + 0U, mean);
        metrics_gm.SetValue(out_base + 1U, std_sample);
        metrics_gm.SetValue(out_base + 2U, std_population);
        metrics_gm.SetValue(out_base + 3U, spec_width / (2.0f * denom_sample));
        metrics_gm.SetValue(out_base + 4U, MinF(cpu, cpl));
        metrics_gm.SetValue(out_base + 5U, spec_width / (2.0f * denom_population));
        metrics_gm.SetValue(out_base + 6U, MinF(ppu, ppl));
        metrics_gm.SetValue(out_base + 7U, harris);
        metrics_gm.SetValue(out_base + 8U, iae);
        metrics_gm.SetValue(out_base + 9U, ise);
        metrics_gm.SetValue(out_base + 10U, itae);
        metrics_gm.SetValue(out_base + 11U, iae / n);
        metrics_gm.SetValue(out_base + 12U, SqrtNewton(ise / MaxF(n * sample_interval, kEpsilon)));
        metrics_gm.SetValue(out_base + 13U, max_abs_error);
        metrics_gm.SetValue(out_base + 14U, out_count / n);
        metrics_gm.SetValue(out_base + 15U, out_count);
        metrics_gm.SetValue(out_base + 16U, MaxF(max_positive, 0.0f) / spec_width);
        metrics_gm.SetValue(out_base + 17U, MaxF(max_negative, 0.0f) / spec_width);
        metrics_gm.SetValue(out_base + 18U, last_unsettled_time);
        metrics_gm.SetValue(out_base + 19U, final_error);
    }
}
