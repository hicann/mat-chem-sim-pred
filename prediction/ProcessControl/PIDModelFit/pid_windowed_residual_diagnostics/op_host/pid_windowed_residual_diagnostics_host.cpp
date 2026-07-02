/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pid_windowed_residual_diagnostics_host.h"

#include <algorithm>
#include <limits>

#include "acl/acl.h"

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

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return ((value + align - 1U) / align) * align;
}

uint32_t ComputeCoreNum(int64_t total_windows)
{
    if (total_windows <= 0) {
        return 1U;
    }
    return static_cast<uint32_t>(std::min<int64_t>(24, total_windows));
}

bool FitsUint32(int64_t value)
{
    constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    return value >= 0 && value <= kUint32Max;
}

bool HasValidShape(int64_t batch, int64_t sample_count, int64_t window_size, int64_t stride, int64_t max_lag)
{
    if (batch <= 0 || sample_count <= 1 || window_size <= 1 || stride <= 0 || max_lag <= 0) {
        return false;
    }
    if (window_size > sample_count || max_lag > window_size - 1) {
        return false;
    }
    if (!FitsUint32(batch) || !FitsUint32(sample_count) || !FitsUint32(window_size) || !FitsUint32(stride) ||
        !FitsUint32(max_lag)) {
        return false;
    }
    const int64_t window_count =
        aclnnPidWindowedResidualDiagnosticsGetWindowCount(sample_count, window_size, stride);
    constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    return window_count > 0 && window_count <= kUint32Max && batch <= kUint32Max / window_count;
}

}  // namespace

extern "C" void aclrtlaunch_pid_windowed_residual_diagnostics_kernel(
    uint32_t blockDim, aclrtStream stream, void* actual, void* predicted, void* metrics, void* autocorr, void* tiling);

extern "C" int32_t aclnnPidWindowedResidualDiagnostics(
    void* actual, void* predicted, void* metrics, void* autocorr, int64_t batch, int64_t sample_count,
    int64_t window_size, int64_t stride, int64_t max_lag, void* workspace, uint64_t workspace_size, void* stream)
{
    if (actual == nullptr || predicted == nullptr || metrics == nullptr || autocorr == nullptr || workspace == nullptr ||
        stream == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (!HasValidShape(batch, sample_count, window_size, stride, max_lag) ||
        workspace_size <
            aclnnPidWindowedResidualDiagnosticsGetWorkspaceSize(batch, sample_count, window_size, stride, max_lag)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    const int64_t window_count =
        aclnnPidWindowedResidualDiagnosticsGetWindowCount(sample_count, window_size, stride);
    const int64_t total_windows = batch * window_count;

    WindowedResidualDiagnosticsTilingData tiling;
    tiling.batch = static_cast<uint32_t>(batch);
    tiling.sample_count = static_cast<uint32_t>(sample_count);
    tiling.window_size = static_cast<uint32_t>(window_size);
    tiling.stride = static_cast<uint32_t>(stride);
    tiling.max_lag = static_cast<uint32_t>(max_lag);
    tiling.window_count = static_cast<uint32_t>(window_count);
    tiling.core_num = ComputeCoreNum(total_windows);
    tiling.metric_count = static_cast<uint32_t>(PID_WINDOWED_RESIDUAL_DIAGNOSTICS_METRIC_COUNT);

    const auto ret = aclrtMemcpyAsync(
        workspace, sizeof(WindowedResidualDiagnosticsTilingData), &tiling,
        sizeof(WindowedResidualDiagnosticsTilingData), ACL_MEMCPY_HOST_TO_DEVICE, reinterpret_cast<aclrtStream>(stream));
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    aclrtlaunch_pid_windowed_residual_diagnostics_kernel(
        tiling.core_num, reinterpret_cast<aclrtStream>(stream), actual, predicted, metrics, autocorr, workspace);
    return ACL_SUCCESS;
}

extern "C" uint64_t aclnnPidWindowedResidualDiagnosticsGetWorkspaceSize(
    int64_t batch, int64_t sample_count, int64_t window_size, int64_t stride, int64_t max_lag)
{
    (void)batch;
    (void)sample_count;
    (void)window_size;
    (void)stride;
    (void)max_lag;
    return AlignUp(sizeof(WindowedResidualDiagnosticsTilingData), 32U);
}

extern "C" int64_t aclnnPidWindowedResidualDiagnosticsGetWindowCount(
    int64_t sample_count, int64_t window_size, int64_t stride)
{
    if (sample_count <= 0 || window_size <= 0 || stride <= 0 || window_size > sample_count) {
        return 0;
    }
    return 1 + (sample_count - window_size) / stride;
}
