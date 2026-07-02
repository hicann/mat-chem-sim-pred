/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PID_FOPDT_BATCH_ROLLOUT_SCORE_HOST_H_
#define PID_FOPDT_BATCH_ROLLOUT_SCORE_HOST_H_

#include <cstdint>

constexpr int64_t PID_FOPDT_BATCH_ROLLOUT_RESULT_COUNT = 8;

extern "C" int32_t aclnnPidFopdtBatchRolloutScore(
    void* a, void* b, void* delay, void* y0, void* sp, void* kp, void* ki, void* kd, void* best_result,
    void* best_idx, int64_t batch, int64_t candidates, int64_t sim_steps, int64_t candidate_tile,
    float sample_interval, float settle_band, float overshoot_weight, float settling_weight, float control_weight,
    void* workspace, uint64_t workspace_size, void* stream);

extern "C" uint64_t aclnnPidFopdtBatchRolloutScoreGetWorkspaceSize(
    int64_t batch, int64_t candidates, int64_t sim_steps, int64_t candidate_tile);

#endif  // PID_FOPDT_BATCH_ROLLOUT_SCORE_HOST_H_
