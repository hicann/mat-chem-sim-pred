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

struct WindowedResidualDiagnosticsTilingData {
    uint32_t batch;
    uint32_t sample_count;
    uint32_t window_size;
    uint32_t stride;
    uint32_t max_lag;
    uint32_t window_count;
    uint32_t core_num;
    uint32_t metric_count;
};

constexpr uint32_t kMetricCount = 8U;
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

__aicore__ inline float AbsF(float value)
{
    return value >= 0.0f ? value : -value;
}

}  // namespace

extern "C" __global__ __aicore__ void pid_windowed_residual_diagnostics_kernel(
    GM_ADDR actual, GM_ADDR predicted, GM_ADDR metrics, GM_ADDR autocorr, GM_ADDR tiling)
{
    const __gm__ WindowedResidualDiagnosticsTilingData* tiling_data =
        reinterpret_cast<const __gm__ WindowedResidualDiagnosticsTilingData*>(tiling);
    const uint32_t batch = tiling_data->batch;
    const uint32_t sample_count = tiling_data->sample_count;
    const uint32_t window_size = tiling_data->window_size;
    const uint32_t stride = tiling_data->stride;
    const uint32_t max_lag = tiling_data->max_lag;
    const uint32_t window_count = tiling_data->window_count;
    const uint32_t core_num = tiling_data->core_num;
    const uint32_t core_idx = GetBlockIdx();
    const uint32_t total_windows = batch * window_count;

    if (batch == 0U || sample_count <= 1U || window_size <= 1U || stride == 0U || max_lag == 0U ||
        max_lag >= window_size || window_count == 0U || core_num == 0U) {
        return;
    }

    GlobalTensor<float> actual_gm;
    GlobalTensor<float> predicted_gm;
    GlobalTensor<float> metrics_gm;
    GlobalTensor<float> autocorr_gm;
    actual_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(actual));
    predicted_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(predicted));
    metrics_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(metrics));
    autocorr_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(autocorr));

    const uint32_t windows_per_core = (total_windows + core_num - 1U) / core_num;
    const uint32_t start_task = core_idx * windows_per_core;
    uint32_t end_task = start_task + windows_per_core;
    if (end_task > total_windows) {
        end_task = total_windows;
    }

    for (uint32_t task = start_task; task < end_task; ++task) {
        const uint32_t loop = task / window_count;
        const uint32_t window = task - loop * window_count;
        const uint32_t window_start = window * stride;
        const uint64_t input_base = static_cast<uint64_t>(loop) * static_cast<uint64_t>(sample_count) + window_start;

        float actual_sum = 0.0f;
        float residual_sum = 0.0f;
        float n = 0.0f;
        for (uint32_t i = 0; i < window_size; ++i) {
            const uint64_t idx = input_base + i;
            const float actual_value = actual_gm.GetValue(idx);
            const float predicted_value = predicted_gm.GetValue(idx);
            n += 1.0f;
            actual_sum += actual_value;
            residual_sum += actual_value - predicted_value;
        }

        const float actual_mean = actual_sum / n;
        const float residual_mean = residual_sum / n;
        float residual_center_energy = 0.0f;
        float actual_energy = 0.0f;
        float sse = 0.0f;
        float abs_sum = 0.0f;
        float max_abs = 0.0f;
        float diff_sum = 0.0f;
        float previous_residual = 0.0f;

        for (uint32_t i = 0; i < window_size; ++i) {
            const uint64_t idx = input_base + i;
            const float actual_value = actual_gm.GetValue(idx);
            const float predicted_value = predicted_gm.GetValue(idx);
            const float residual = actual_value - predicted_value;
            const float centered_residual = residual - residual_mean;
            const float centered_actual = actual_value - actual_mean;
            const float abs_residual = AbsF(residual);
            residual_center_energy += centered_residual * centered_residual;
            actual_energy += centered_actual * centered_actual;
            sse += residual * residual;
            abs_sum += abs_residual;
            max_abs = MaxF(max_abs, abs_residual);
            if (i > 0U) {
                const float diff = residual - previous_residual;
                diff_sum += diff * diff;
            }
            previous_residual = residual;
        }

        float ljung_box_q = 0.0f;
        const uint64_t autocorr_base =
            (static_cast<uint64_t>(loop) * static_cast<uint64_t>(window_count) + window) *
            static_cast<uint64_t>(max_lag);
        for (uint32_t lag = 1U; lag <= max_lag; ++lag) {
            float numerator = 0.0f;
            float lag_count = 0.0f;
            for (uint32_t i = lag; i < window_size; ++i) {
                const uint64_t lhs_idx = input_base + i;
                const uint64_t rhs_idx = input_base + i - lag;
                const float residual_lhs = actual_gm.GetValue(lhs_idx) - predicted_gm.GetValue(lhs_idx);
                const float residual_rhs = actual_gm.GetValue(rhs_idx) - predicted_gm.GetValue(rhs_idx);
                lag_count += 1.0f;
                numerator += (residual_lhs - residual_mean) * (residual_rhs - residual_mean);
            }
            const float value = numerator / MaxF(residual_center_energy, kEpsilon);
            autocorr_gm.SetValue(autocorr_base + lag - 1U, value);
            ljung_box_q += value * value / MaxF(lag_count, 1.0f);
        }
        ljung_box_q *= n * (n + 2.0f);

        const float rmse = SqrtNewton(sse / n);
        const float actual_norm = SqrtNewton(MaxF(actual_energy, kEpsilon));
        const float residual_norm = SqrtNewton(sse);
        const float residual_ratio = residual_norm / actual_norm;
        const float fit_percent = 100.0f * (1.0f - residual_ratio);
        const float durbin_watson = diff_sum / MaxF(sse, kEpsilon);
        const uint64_t out_base =
            (static_cast<uint64_t>(loop) * static_cast<uint64_t>(window_count) + window) * kMetricCount;
        metrics_gm.SetValue(out_base + 0U, residual_mean);
        metrics_gm.SetValue(out_base + 1U, SqrtNewton(residual_center_energy / MaxF(n - 1.0f, 1.0f)));
        metrics_gm.SetValue(out_base + 2U, abs_sum / n);
        metrics_gm.SetValue(out_base + 3U, rmse);
        metrics_gm.SetValue(out_base + 4U, max_abs);
        metrics_gm.SetValue(out_base + 5U, fit_percent);
        metrics_gm.SetValue(out_base + 6U, durbin_watson);
        metrics_gm.SetValue(out_base + 7U, ljung_box_q);
    }
}
