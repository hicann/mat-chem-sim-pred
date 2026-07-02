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

struct ProcessCapabilityTilingData {
    uint32_t batch;
    uint32_t sample_count;
    uint32_t core_num;
    uint32_t metric_count;
};

constexpr uint32_t kMetricCount = 13U;
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

}  // namespace

extern "C" __global__ __aicore__ void pid_process_capability_metrics_kernel(
    GM_ADDR values, GM_ADDR lsl, GM_ADDR usl, GM_ADDR metrics, GM_ADDR tiling)
{
    const __gm__ ProcessCapabilityTilingData* tiling_data =
        reinterpret_cast<const __gm__ ProcessCapabilityTilingData*>(tiling);
    const uint32_t batch = tiling_data->batch;
    const uint32_t sample_count = tiling_data->sample_count;
    const uint32_t core_num = tiling_data->core_num;
    const uint32_t core_idx = GetBlockIdx();

    if (batch == 0U || sample_count == 0U || core_num == 0U) {
        return;
    }

    GlobalTensor<float> values_gm;
    GlobalTensor<float> lsl_gm;
    GlobalTensor<float> usl_gm;
    GlobalTensor<float> metrics_gm;
    values_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values));
    lsl_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lsl));
    usl_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(usl));
    metrics_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(metrics));

    const uint32_t loops_per_core = (batch + core_num - 1U) / core_num;
    const uint32_t start_loop = core_idx * loops_per_core;
    uint32_t end_loop = start_loop + loops_per_core;
    if (end_loop > batch) {
        end_loop = batch;
    }

    for (uint32_t loop = start_loop; loop < end_loop; ++loop) {
        float mean = 0.0f;
        float m2 = 0.0f;
        float n = 0.0f;
        float out_count = 0.0f;
        const float lower = lsl_gm.GetValue(loop);
        const float upper = usl_gm.GetValue(loop);
        const uint64_t base = static_cast<uint64_t>(loop) * sample_count;
        float min_value = values_gm.GetValue(base);
        float max_value = min_value;

        for (uint32_t i = 0; i < sample_count; ++i) {
            const float value = values_gm.GetValue(base + i);
            n += 1.0f;
            const float delta = value - mean;
            mean += delta / n;
            const float delta2 = value - mean;
            m2 += delta * delta2;
            min_value = MinF(min_value, value);
            max_value = MaxF(max_value, value);
            if (value < lower || value > upper) {
                out_count += 1.0f;
            }
        }

        const float var_population = MaxF(0.0f, m2 / n);
        const float var_sample = sample_count > 1U ? m2 / (n - 1.0f) : var_population;
        const float std_population = SqrtNewton(var_population);
        const float std_sample = SqrtNewton(var_sample);
        const float denom_sample = 3.0f * MaxF(std_sample, kEpsilon);
        const float denom_population = 3.0f * MaxF(std_population, kEpsilon);
        const float spec_width = upper - lower;
        const float cpu = (upper - mean) / denom_sample;
        const float cpl = (mean - lower) / denom_sample;
        const float ppu = (upper - mean) / denom_population;
        const float ppl = (mean - lower) / denom_population;
        const uint64_t out_base = static_cast<uint64_t>(loop) * kMetricCount;

        metrics_gm.SetValue(out_base + 0U, mean);
        metrics_gm.SetValue(out_base + 1U, std_sample);
        metrics_gm.SetValue(out_base + 2U, std_population);
        metrics_gm.SetValue(out_base + 3U, spec_width / (2.0f * denom_sample));
        metrics_gm.SetValue(out_base + 4U, cpu);
        metrics_gm.SetValue(out_base + 5U, cpl);
        metrics_gm.SetValue(out_base + 6U, MinF(cpu, cpl));
        metrics_gm.SetValue(out_base + 7U, spec_width / (2.0f * denom_population));
        metrics_gm.SetValue(out_base + 8U, MinF(ppu, ppl));
        metrics_gm.SetValue(out_base + 9U, out_count / n);
        metrics_gm.SetValue(out_base + 10U, out_count);
        metrics_gm.SetValue(out_base + 11U, min_value);
        metrics_gm.SetValue(out_base + 12U, max_value);
    }
}
