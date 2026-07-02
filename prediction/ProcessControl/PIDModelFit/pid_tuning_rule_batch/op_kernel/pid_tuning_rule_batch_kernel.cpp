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

struct TuningRuleBatchTilingData {
    uint32_t batch;
    uint32_t core_num;
    uint32_t rule_count;
    uint32_t param_count;
    uint32_t diagnostic_count;
};

constexpr uint32_t kRuleCount = 3U;
constexpr uint32_t kParamCount = 3U;
constexpr uint32_t kDiagnosticCount = 4U;
constexpr float kEpsilon = 1.0e-6f;

__aicore__ inline float AbsF(float value)
{
    return value >= 0.0f ? value : -value;
}

__aicore__ inline float MaxF(float lhs, float rhs)
{
    return lhs > rhs ? lhs : rhs;
}

__aicore__ inline void StoreRule(
    GlobalTensor<float>& pid_params_gm, GlobalTensor<float>& diagnostics_gm, uint32_t loop, uint32_t rule, float kp,
    float ki, float kd, float tau, float valid, float ratio, float lambda_ratio)
{
    const uint64_t param_base = (static_cast<uint64_t>(loop) * kRuleCount + rule) * kParamCount;
    const uint64_t diag_base = (static_cast<uint64_t>(loop) * kRuleCount + rule) * kDiagnosticCount;
    const float aggressiveness = valid > 0.5f ? AbsF(kp) + tau * AbsF(ki) + AbsF(kd) / MaxF(tau, kEpsilon) : 0.0f;
    pid_params_gm.SetValue(param_base + 0U, valid > 0.5f ? kp : 0.0f);
    pid_params_gm.SetValue(param_base + 1U, valid > 0.5f ? ki : 0.0f);
    pid_params_gm.SetValue(param_base + 2U, valid > 0.5f ? kd : 0.0f);
    diagnostics_gm.SetValue(diag_base + 0U, valid);
    diagnostics_gm.SetValue(diag_base + 1U, ratio);
    diagnostics_gm.SetValue(diag_base + 2U, aggressiveness);
    diagnostics_gm.SetValue(diag_base + 3U, lambda_ratio);
}

}  // namespace

extern "C" __global__ __aicore__ void pid_tuning_rule_batch_kernel(
    GM_ADDR process_gain, GM_ADDR time_constant, GM_ADDR dead_time, GM_ADDR lambda_value, GM_ADDR pid_params,
    GM_ADDR diagnostics, GM_ADDR tiling)
{
    const __gm__ TuningRuleBatchTilingData* tiling_data =
        reinterpret_cast<const __gm__ TuningRuleBatchTilingData*>(tiling);
    const uint32_t batch = tiling_data->batch;
    const uint32_t core_num = tiling_data->core_num;
    const uint32_t core_idx = GetBlockIdx();

    if (batch == 0U || core_num == 0U) {
        return;
    }

    GlobalTensor<float> gain_gm;
    GlobalTensor<float> tau_gm;
    GlobalTensor<float> theta_gm;
    GlobalTensor<float> lambda_gm;
    GlobalTensor<float> pid_params_gm;
    GlobalTensor<float> diagnostics_gm;
    gain_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(process_gain));
    tau_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(time_constant));
    theta_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dead_time));
    lambda_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lambda_value));
    pid_params_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pid_params));
    diagnostics_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diagnostics));

    const uint32_t loops_per_core = (batch + core_num - 1U) / core_num;
    const uint32_t start_loop = core_idx * loops_per_core;
    uint32_t end_loop = start_loop + loops_per_core;
    if (end_loop > batch) {
        end_loop = batch;
    }

    for (uint32_t loop = start_loop; loop < end_loop; ++loop) {
        const float gain = gain_gm.GetValue(loop);
        const float tau_raw = tau_gm.GetValue(loop);
        const float theta_raw = theta_gm.GetValue(loop);
        const float lambda_raw = lambda_gm.GetValue(loop);
        const float valid =
            (AbsF(gain) > kEpsilon && tau_raw > kEpsilon && theta_raw > kEpsilon && lambda_raw > kEpsilon) ? 1.0f
                                                                                                           : 0.0f;
        const float safe_gain = AbsF(gain) > kEpsilon ? gain : 1.0f;
        const float tau = MaxF(tau_raw, kEpsilon);
        const float theta = MaxF(theta_raw, kEpsilon);
        const float lambda = MaxF(lambda_raw, kEpsilon);
        const float ratio = theta / tau;
        const float lambda_ratio = lambda / theta;

        float kp = 1.2f * tau / (safe_gain * theta);
        float ti = 2.0f * theta;
        float td = 0.5f * theta;
        StoreRule(pid_params_gm, diagnostics_gm, loop, 0U, kp, kp / MaxF(ti, kEpsilon), kp * td, tau, valid, ratio,
                  lambda_ratio);

        kp = (tau + 0.5f * theta) / (safe_gain * (lambda + 0.5f * theta));
        ti = tau + 0.5f * theta;
        td = tau * theta / MaxF(2.0f * tau + theta, kEpsilon);
        StoreRule(pid_params_gm, diagnostics_gm, loop, 1U, kp, kp / MaxF(ti, kEpsilon), kp * td, tau, valid, ratio,
                  lambda_ratio);

        kp = (tau / (safe_gain * theta)) * (1.3333333333333333f + theta / (4.0f * tau));
        ti = theta * (32.0f + 6.0f * ratio) / MaxF(13.0f + 8.0f * ratio, kEpsilon);
        td = theta * 4.0f / MaxF(11.0f + 2.0f * ratio, kEpsilon);
        StoreRule(pid_params_gm, diagnostics_gm, loop, 2U, kp, kp / MaxF(ti, kEpsilon), kp * td, tau, valid, ratio,
                  lambda_ratio);
    }
}
