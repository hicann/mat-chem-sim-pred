/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pid_step_response_features_host.h"

#include <algorithm>
#include <limits>

#include "acl/acl.h"

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

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return ((value + align - 1U) / align) * align;
}

uint32_t ComputeCoreNum(int64_t total_tasks)
{
    if (total_tasks <= 0) {
        return 1U;
    }
    return static_cast<uint32_t>(std::min<int64_t>(24, total_tasks));
}

bool FitsUint32(int64_t value)
{
    constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    return value >= 0 && value <= kUint32Max;
}

bool HasValidShape(int64_t batch, int64_t candidates, int64_t sample_count)
{
    if (batch <= 0 || candidates <= 0 || sample_count <= 1) {
        return false;
    }
    if (!FitsUint32(batch) || !FitsUint32(candidates) || !FitsUint32(sample_count)) {
        return false;
    }
    constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    return batch <= kUint32Max / candidates;
}

}  // namespace

extern "C" void aclrtlaunch_pid_step_response_features_kernel(
    uint32_t blockDim, aclrtStream stream, void* pv_candidates, void* sp, void* features, void* tiling);

extern "C" int32_t aclnnPidStepResponseFeatures(
    void* pv_candidates, void* sp, void* features, int64_t batch, int64_t candidates, int64_t sample_count,
    float sample_interval, float settle_band_ratio, void* workspace, uint64_t workspace_size, void* stream)
{
    if (pv_candidates == nullptr || sp == nullptr || features == nullptr || workspace == nullptr || stream == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (!HasValidShape(batch, candidates, sample_count) ||
        workspace_size < aclnnPidStepResponseFeaturesGetWorkspaceSize(batch, candidates, sample_count) ||
        sample_interval <= 0.0f || settle_band_ratio < 0.0f) {
        return ACL_ERROR_INVALID_PARAM;
    }

    const int64_t total_tasks = batch * candidates;
    StepResponseFeaturesTilingData tiling;
    tiling.batch = static_cast<uint32_t>(batch);
    tiling.candidates = static_cast<uint32_t>(candidates);
    tiling.sample_count = static_cast<uint32_t>(sample_count);
    tiling.total_tasks = static_cast<uint32_t>(total_tasks);
    tiling.core_num = ComputeCoreNum(total_tasks);
    tiling.feature_count = static_cast<uint32_t>(PID_STEP_RESPONSE_FEATURE_COUNT);
    tiling.sample_interval = sample_interval;
    tiling.settle_band_ratio = settle_band_ratio;

    const auto ret = aclrtMemcpyAsync(
        workspace, sizeof(StepResponseFeaturesTilingData), &tiling, sizeof(StepResponseFeaturesTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE, reinterpret_cast<aclrtStream>(stream));
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    aclrtlaunch_pid_step_response_features_kernel(
        tiling.core_num, reinterpret_cast<aclrtStream>(stream), pv_candidates, sp, features, workspace);
    return ACL_SUCCESS;
}

extern "C" uint64_t aclnnPidStepResponseFeaturesGetWorkspaceSize(
    int64_t batch, int64_t candidates, int64_t sample_count)
{
    (void)batch;
    (void)candidates;
    (void)sample_count;
    return AlignUp(sizeof(StepResponseFeaturesTilingData), 32U);
}
