/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pid_fopdt_batch_rollout_score_host.h"

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"

namespace {

struct LocalTilingData {
    uint32_t batch;
    uint32_t candidates;
    uint32_t sim_steps;
    uint32_t candidate_start;
    uint32_t candidate_count;
    uint32_t tile_index;
    uint32_t tile_count;
    uint32_t core_num;
    uint32_t reserved;
    float sample_interval;
    float settle_band;
    float overshoot_weight;
    float settling_weight;
    float control_weight;
};

struct FinalTilingData {
    uint32_t batch;
    uint32_t candidates;
    uint32_t core_num;
    uint32_t reserved;
};

constexpr uint32_t kMetricPlanes = 5U;  // score, iae, ise, overshoot, settling

constexpr uint32_t kStableCoreNum = 24U;
constexpr uint32_t kLoopsPerTaskUnit = 1U;

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return ((value + align - 1U) / align) * align;
}

uint32_t ComputeCoreNum(int64_t batch)
{
    if (batch <= 0) {
        return 1U;
    }
    const uint32_t task_units = (static_cast<uint32_t>(batch) + kLoopsPerTaskUnit - 1U) / kLoopsPerTaskUnit;
    return task_units > kStableCoreNum ? kStableCoreNum : (task_units == 0U ? 1U : task_units);
}

uint64_t ComputeTileCount(int64_t candidates, int64_t candidate_tile)
{
    if (candidates <= 0 || candidate_tile <= 0) {
        return 0U;
    }
    const uint64_t tile = static_cast<uint64_t>(std::min<int64_t>(candidate_tile, candidates));
    return (static_cast<uint64_t>(candidates) + tile - 1U) / tile;
}

int64_t ResolveCandidateTile(int64_t candidates, int64_t candidate_tile)
{
    constexpr int64_t kDefaultCandidateLane = 768;  // = kLane SIMD width; optimal default tile
    if (candidate_tile > 0) {
        return candidate_tile;
    }
    return std::min<int64_t>(candidates, kDefaultCandidateLane);
}

uint64_t ComputeTilingStride()
{
    return AlignUp(std::max<uint64_t>(sizeof(LocalTilingData), sizeof(FinalTilingData)), 32U);
}

}  // namespace

extern "C" uint32_t aclrtlaunch_pid_fopdt_batch_rollout_score_local_kernel(
    uint32_t blockDim, aclrtStream stream, void* a, void* b, void* delay, void* y0, void* sp, void* kp, void* ki,
    void* kd, void* partial_result, void* partial_idx, void* tiling);

extern "C" uint32_t aclrtlaunch_pid_fopdt_batch_rollout_score_final_kernel(
    uint32_t blockDim, aclrtStream stream, void* kp, void* ki, void* kd, void* metrics, void* best_result,
    void* best_idx, void* tiling);

static uint64_t PlaneStride(int64_t batch, int64_t candidates)
{
    return AlignUp(static_cast<uint64_t>(batch) * static_cast<uint64_t>(candidates), 8U);
}

extern "C" int32_t aclnnPidFopdtBatchRolloutScore(
    void* a, void* b, void* delay, void* y0, void* sp, void* kp, void* ki, void* kd, void* best_result,
    void* best_idx, int64_t batch, int64_t candidates, int64_t sim_steps, int64_t candidate_tile,
    float sample_interval, float settle_band, float overshoot_weight, float settling_weight, float control_weight,
    void* workspace, uint64_t workspace_size, void* stream)
{
    if (a == nullptr || b == nullptr || delay == nullptr || y0 == nullptr || sp == nullptr || kp == nullptr ||
        ki == nullptr || kd == nullptr || best_result == nullptr || best_idx == nullptr || workspace == nullptr ||
        stream == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (batch <= 0 || candidates <= 0 || sim_steps <= 0 ||
        workspace_size < aclnnPidFopdtBatchRolloutScoreGetWorkspaceSize(batch, candidates, sim_steps, candidate_tile)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    candidate_tile = ResolveCandidateTile(candidates, candidate_tile);

    const uint64_t tile_count = ComputeTileCount(candidates, candidate_tile);
    const uint64_t metrics_bytes =
        AlignUp(kMetricPlanes * PlaneStride(batch, candidates) * sizeof(float), 32U);
    const uint64_t tiling_stride = ComputeTilingStride();

    uint8_t* workspace_bytes = reinterpret_cast<uint8_t*>(workspace);
    void* metrics = workspace_bytes;
    uint8_t* tiling_base = workspace_bytes + metrics_bytes;
    const uint32_t core_num = ComputeCoreNum(batch);

    for (uint64_t tile = 0U; tile < tile_count; ++tile) {
        const uint64_t candidate_start = tile * static_cast<uint64_t>(candidate_tile);
        const uint64_t remaining = static_cast<uint64_t>(candidates) - candidate_start;
        const uint64_t candidate_count = std::min<uint64_t>(static_cast<uint64_t>(candidate_tile), remaining);

        LocalTilingData local_tiling;
        local_tiling.batch = static_cast<uint32_t>(batch);
        local_tiling.candidates = static_cast<uint32_t>(candidates);
        local_tiling.sim_steps = static_cast<uint32_t>(sim_steps);
        local_tiling.candidate_start = static_cast<uint32_t>(candidate_start);
        local_tiling.candidate_count = static_cast<uint32_t>(candidate_count);
        local_tiling.tile_index = static_cast<uint32_t>(tile);
        local_tiling.tile_count = static_cast<uint32_t>(tile_count);
        local_tiling.core_num = core_num;
        local_tiling.reserved = 0U;
        local_tiling.sample_interval = sample_interval;
        local_tiling.settle_band = settle_band;
        local_tiling.overshoot_weight = overshoot_weight;
        local_tiling.settling_weight = settling_weight;
        local_tiling.control_weight = control_weight;

        void* local_tiling_device = tiling_base + tile * tiling_stride;
        auto ret = aclrtMemcpy(
            local_tiling_device, sizeof(LocalTilingData), &local_tiling, sizeof(LocalTilingData),
            ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            return ret;
        }

        const uint32_t launch_ret = aclrtlaunch_pid_fopdt_batch_rollout_score_local_kernel(
            core_num, reinterpret_cast<aclrtStream>(stream), a, b, delay, y0, sp, kp, ki, kd, metrics,
            metrics, local_tiling_device);
        if (launch_ret != ACL_SUCCESS) {
            return static_cast<int32_t>(launch_ret);
        }
    }

    FinalTilingData final_tiling;
    final_tiling.batch = static_cast<uint32_t>(batch);
    final_tiling.candidates = static_cast<uint32_t>(candidates);
    final_tiling.core_num = core_num;
    final_tiling.reserved = 0U;
    void* final_tiling_device = tiling_base + tile_count * tiling_stride;
    const auto ret = aclrtMemcpy(
        final_tiling_device, sizeof(FinalTilingData), &final_tiling, sizeof(FinalTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const uint32_t launch_ret = aclrtlaunch_pid_fopdt_batch_rollout_score_final_kernel(
        core_num, reinterpret_cast<aclrtStream>(stream), kp, ki, kd, metrics, best_result, best_idx,
        final_tiling_device);
    return static_cast<int32_t>(launch_ret);
}

extern "C" uint64_t aclnnPidFopdtBatchRolloutScoreGetWorkspaceSize(
    int64_t batch, int64_t candidates, int64_t sim_steps, int64_t candidate_tile)
{
    (void)sim_steps;
    if (batch <= 0 || candidates <= 0) {
        return 0U;
    }
    candidate_tile = ResolveCandidateTile(candidates, candidate_tile);
    const uint64_t tile_count = ComputeTileCount(candidates, candidate_tile);
    const uint64_t metrics_bytes =
        AlignUp(kMetricPlanes * PlaneStride(batch, candidates) * sizeof(float), 32U);
    const uint64_t tiling_bytes = (tile_count + 1U) * ComputeTilingStride();
    return metrics_bytes + tiling_bytes;
}
