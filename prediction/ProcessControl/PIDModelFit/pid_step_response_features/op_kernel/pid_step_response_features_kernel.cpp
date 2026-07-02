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

struct StepResponseFeaturesTilingData {
    uint32_t batch;
    uint32_t candidates;
    uint32_t sample_count;
    uint32_t total_tasks;
    uint32_t core_num;
    uint32_t feature_count;
    float sample_interval;
    float settle_band_ratio;
};

constexpr uint32_t kFeatureCount = 12U;
constexpr float kEpsilon = 1.0e-6f;

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

extern "C" __global__ __aicore__ void pid_step_response_features_kernel(
    GM_ADDR pv_candidates, GM_ADDR sp, GM_ADDR features, GM_ADDR tiling)
{
    const __gm__ StepResponseFeaturesTilingData* tiling_data =
        reinterpret_cast<const __gm__ StepResponseFeaturesTilingData*>(tiling);
    const uint32_t batch = tiling_data->batch;
    const uint32_t candidates = tiling_data->candidates;
    const uint32_t sample_count = tiling_data->sample_count;
    const uint32_t total_tasks = tiling_data->total_tasks;
    const uint32_t core_num = tiling_data->core_num;
    const float sample_interval = tiling_data->sample_interval;
    const float settle_band_ratio = tiling_data->settle_band_ratio;
    const uint32_t core_idx = GetBlockIdx();

    if (batch == 0U || candidates == 0U || sample_count <= 1U || total_tasks == 0U || core_num == 0U ||
        sample_interval <= 0.0f) {
        return;
    }

    GlobalTensor<float> pv_gm;
    GlobalTensor<float> sp_gm;
    GlobalTensor<float> features_gm;
    pv_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pv_candidates));
    sp_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sp));
    features_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(features));

    const uint32_t tasks_per_core = (total_tasks + core_num - 1U) / core_num;
    const uint32_t start_task = core_idx * tasks_per_core;
    uint32_t end_task = start_task + tasks_per_core;
    if (end_task > total_tasks) {
        end_task = total_tasks;
    }

    for (uint32_t task = start_task; task < end_task; ++task) {
        const uint32_t loop = task / candidates;
        const uint32_t candidate = task - loop * candidates;
        const uint64_t pv_base =
            (static_cast<uint64_t>(loop) * static_cast<uint64_t>(candidates) + candidate) *
            static_cast<uint64_t>(sample_count);
        const uint64_t sp_base = static_cast<uint64_t>(loop) * static_cast<uint64_t>(sample_count);

        const float target = sp_gm.GetValue(sp_base + sample_count - 1U);
        const float initial = pv_gm.GetValue(pv_base);
        const float final_value = pv_gm.GetValue(pv_base + sample_count - 1U);
        const float delta = target - initial;
        const float abs_delta = MaxF(AbsF(delta), kEpsilon);
        const float direction = delta >= 0.0f ? 1.0f : -1.0f;
        const float band = MaxF(abs_delta * settle_band_ratio, 1.0e-4f);

        float peak = initial;
        float trough = initial;
        float peak_time = 0.0f;
        float rise10_time = 0.0f;
        float rise90_time = 0.0f;
        float last_time = 0.0f;
        bool found10 = false;
        bool found90 = false;
        float settling_time = 0.0f;
        float iae = 0.0f;
        float ise = 0.0f;
        float time_value = 0.0f;

        for (uint32_t i = 0; i < sample_count; ++i) {
            const float pv_value = pv_gm.GetValue(pv_base + i);
            const float sp_value = sp_gm.GetValue(sp_base + i);
            if (pv_value > peak) {
                peak = pv_value;
                peak_time = time_value;
            }
            trough = MinF(trough, pv_value);

            const float normalized = direction * (pv_value - initial) / abs_delta;
            if (!found10 && normalized >= 0.1f) {
                rise10_time = time_value;
                found10 = true;
            }
            if (!found90 && normalized >= 0.9f) {
                rise90_time = time_value;
                found90 = true;
            }

            const float error = AbsF(sp_value - pv_value);
            iae += error * sample_interval;
            ise += error * error * sample_interval;
            if (error > band) {
                settling_time = time_value + sample_interval;
            }
            last_time = time_value;
            time_value += sample_interval;
        }

        if (!found10) {
            rise10_time = last_time;
        }
        if (!found90) {
            rise90_time = last_time;
        }
        const float rise_time = rise90_time >= rise10_time ? rise90_time - rise10_time : 0.0f;
        const float overshoot = MaxF(direction * (peak - target), 0.0f) / abs_delta;
        const float undershoot = MaxF(direction * (target - trough), 0.0f) / abs_delta;
        const uint64_t out_base = static_cast<uint64_t>(task) * kFeatureCount;

        features_gm.SetValue(out_base + 0U, initial);
        features_gm.SetValue(out_base + 1U, final_value);
        features_gm.SetValue(out_base + 2U, AbsF(target - final_value));
        features_gm.SetValue(out_base + 3U, peak);
        features_gm.SetValue(out_base + 4U, trough);
        features_gm.SetValue(out_base + 5U, overshoot);
        features_gm.SetValue(out_base + 6U, undershoot);
        features_gm.SetValue(out_base + 7U, rise_time);
        features_gm.SetValue(out_base + 8U, peak_time);
        features_gm.SetValue(out_base + 9U, settling_time);
        features_gm.SetValue(out_base + 10U, iae);
        features_gm.SetValue(out_base + 11U, ise);
    }
}
