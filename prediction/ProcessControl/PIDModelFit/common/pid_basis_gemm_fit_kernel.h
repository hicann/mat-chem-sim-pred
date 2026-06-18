/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PID_BASIS_GEMM_FIT_KERNEL_H_
#define PID_BASIS_GEMM_FIT_KERNEL_H_

#include "kernel_operator.h"
#include "pid_basis_gemm_fit_common.h"

namespace pid_basis_gemm {

using namespace AscendC;

constexpr float kEpsilon = 1.0e-6f;
constexpr float kLargeSse = 3.4028234663852886e38f;
constexpr uint32_t kLoopsPerTaskUnit = 16U;

__aicore__ inline float AbsF(float value)
{
    return value < 0.0f ? -value : value;
}

__aicore__ inline bool IsInvalidScore(float value)
{
    return value != value || AbsF(value) > 1.0e30f;
}

class PidBasisGemmFitReduceOp {
public:
    __aicore__ inline PidBasisGemmFitReduceOp() {}

    __aicore__ inline void Init(
        GM_ADDR dot, GM_ADDR basis_norm, GM_ADDR y_energy, GM_ADDR best_sse, GM_ADDR best_k, GM_ADDR best_idx,
        GM_ADDR tiling)
    {
        const __gm__ PidBasisGemmFitTilingData* tiling_data =
            reinterpret_cast<const __gm__ PidBasisGemmFitTilingData*>(tiling);
        batch_ = tiling_data->batch;
        candidates_ = tiling_data->candidates;
        core_num_ = tiling_data->core_num;
        core_idx_ = GetBlockIdx();

        dot_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dot));
        basis_norm_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(basis_norm));
        y_energy_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y_energy));
        best_sse_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(best_sse));
        best_k_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(best_k));
        best_idx_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(best_idx));
    }

    __aicore__ inline void Process()
    {
        if (batch_ == 0U || candidates_ == 0U || core_num_ == 0U) {
            return;
        }

        const uint32_t task_units = (batch_ + kLoopsPerTaskUnit - 1U) / kLoopsPerTaskUnit;
        const uint32_t units_per_core = (task_units + core_num_ - 1U) / core_num_;
        const uint32_t start_unit = core_idx_ * units_per_core;
        uint32_t end_unit = start_unit + units_per_core;
        if (end_unit > task_units) {
            end_unit = task_units;
        }

        const uint32_t start_loop = start_unit * kLoopsPerTaskUnit;
        uint32_t end_loop = end_unit * kLoopsPerTaskUnit;
        if (end_loop > batch_) {
            end_loop = batch_;
        }

        for (uint32_t loop = start_loop; loop < end_loop; ++loop) {
            ReduceOneLoop(loop);
        }
    }

private:
    __aicore__ inline void ReduceOneLoop(uint32_t loop)
    {
        const float energy = y_energy_gm_.GetValue(loop);
        float best_score = kLargeSse;
        float best_gain = 0.0f;
        int32_t best_candidate = 0;

        for (uint32_t candidate = 0; candidate < candidates_; ++candidate) {
            const float norm = basis_norm_gm_.GetValue(candidate);
            const float dot_value = dot_gm_.GetValue(static_cast<uint64_t>(loop) * candidates_ + candidate);
            float gain = 0.0f;
            float score = energy;
            if (norm > kEpsilon) {
                gain = dot_value / norm;
                score = energy - dot_value * dot_value / norm;
                if (score < 0.0f && score > -1.0e-3f) {
                    score = 0.0f;
                }
            }
            if (IsInvalidScore(score)) {
                score = kLargeSse;
                gain = 0.0f;
            }
            if (score < best_score) {
                best_score = score;
                best_gain = gain;
                best_candidate = static_cast<int32_t>(candidate);
            }
        }

        best_sse_gm_.SetValue(loop, best_score);
        best_k_gm_.SetValue(loop, best_gain);
        best_idx_gm_.SetValue(loop, best_candidate);
    }

    GlobalTensor<float> dot_gm_;
    GlobalTensor<float> basis_norm_gm_;
    GlobalTensor<float> y_energy_gm_;
    GlobalTensor<float> best_sse_gm_;
    GlobalTensor<float> best_k_gm_;
    GlobalTensor<int32_t> best_idx_gm_;
    uint32_t batch_ = 0U;
    uint32_t candidates_ = 0U;
    uint32_t core_num_ = 1U;
    uint32_t core_idx_ = 0U;
};

}  // namespace pid_basis_gemm

#endif  // PID_BASIS_GEMM_FIT_KERNEL_H_
