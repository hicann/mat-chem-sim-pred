/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PID_STEP_RESPONSE_FEATURES_HOST_H
#define PID_STEP_RESPONSE_FEATURES_HOST_H

#include <cstdint>

constexpr int64_t PID_STEP_RESPONSE_FEATURE_COUNT = 12;

extern "C" int32_t aclnnPidStepResponseFeatures(
    void* pv_candidates, void* sp, void* features, int64_t batch, int64_t candidates, int64_t sample_count,
    float sample_interval, float settle_band_ratio, void* workspace, uint64_t workspace_size, void* stream);

extern "C" uint64_t aclnnPidStepResponseFeaturesGetWorkspaceSize(
    int64_t batch, int64_t candidates, int64_t sample_count);

#endif  // PID_STEP_RESPONSE_FEATURES_HOST_H
