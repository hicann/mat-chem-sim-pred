/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PID_WINDOWED_RESIDUAL_DIAGNOSTICS_HOST_H
#define PID_WINDOWED_RESIDUAL_DIAGNOSTICS_HOST_H

#include <cstdint>

constexpr int64_t PID_WINDOWED_RESIDUAL_DIAGNOSTICS_METRIC_COUNT = 8;

extern "C" int32_t aclnnPidWindowedResidualDiagnostics(
    void* actual, void* predicted, void* metrics, void* autocorr, int64_t batch, int64_t sample_count,
    int64_t window_size, int64_t stride, int64_t max_lag, void* workspace, uint64_t workspace_size, void* stream);

extern "C" uint64_t aclnnPidWindowedResidualDiagnosticsGetWorkspaceSize(
    int64_t batch, int64_t sample_count, int64_t window_size, int64_t stride, int64_t max_lag);

extern "C" int64_t aclnnPidWindowedResidualDiagnosticsGetWindowCount(
    int64_t sample_count, int64_t window_size, int64_t stride);

#endif  // PID_WINDOWED_RESIDUAL_DIAGNOSTICS_HOST_H
