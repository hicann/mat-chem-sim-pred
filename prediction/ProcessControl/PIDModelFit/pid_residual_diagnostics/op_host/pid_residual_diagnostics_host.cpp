/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pid_residual_diagnostics_host.h"

#include <algorithm>
#include <limits>

#include "acl/acl.h"

namespace {

struct ResidualDiagnosticsTilingData {
    uint32_t batch;
    uint32_t sample_count;
    uint32_t max_lag;
    uint32_t core_num;
    uint32_t metric_count;
};

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return ((value + align - 1U) / align) * align;
}

uint32_t ComputeCoreNum(int64_t batch)
{
    if (batch <= 0) {
        return 1U;
    }
    return static_cast<uint32_t>(std::min<int64_t>(24, batch));
}

bool HasValidShape(int64_t batch, int64_t sample_count, int64_t max_lag)
{
    constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    return batch > 0 && sample_count > 1 && max_lag > 0 && batch <= kUint32Max && sample_count <= kUint32Max &&
           max_lag <= sample_count - 1 && max_lag <= kUint32Max;
}

}  // namespace

extern "C" void aclrtlaunch_pid_residual_diagnostics_kernel(
    uint32_t blockDim, aclrtStream stream, void* actual, void* predicted, void* metrics, void* autocorr, void* tiling);

extern "C" int32_t aclnnPidResidualDiagnostics(
    void* actual, void* predicted, void* metrics, void* autocorr, int64_t batch, int64_t sample_count,
    int64_t max_lag, void* workspace, uint64_t workspace_size, void* stream)
{
    if (actual == nullptr || predicted == nullptr || metrics == nullptr || autocorr == nullptr || workspace == nullptr ||
        stream == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (!HasValidShape(batch, sample_count, max_lag) ||
        workspace_size < aclnnPidResidualDiagnosticsGetWorkspaceSize(batch, sample_count, max_lag)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    ResidualDiagnosticsTilingData tiling;
    tiling.batch = static_cast<uint32_t>(batch);
    tiling.sample_count = static_cast<uint32_t>(sample_count);
    tiling.max_lag = static_cast<uint32_t>(max_lag);
    tiling.core_num = ComputeCoreNum(batch);
    tiling.metric_count = static_cast<uint32_t>(PID_RESIDUAL_DIAGNOSTICS_METRIC_COUNT);

    const auto ret = aclrtMemcpyAsync(
        workspace, sizeof(ResidualDiagnosticsTilingData), &tiling, sizeof(ResidualDiagnosticsTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE, reinterpret_cast<aclrtStream>(stream));
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    aclrtlaunch_pid_residual_diagnostics_kernel(
        tiling.core_num, reinterpret_cast<aclrtStream>(stream), actual, predicted, metrics, autocorr, workspace);
    return ACL_SUCCESS;
}

extern "C" uint64_t aclnnPidResidualDiagnosticsGetWorkspaceSize(
    int64_t batch, int64_t sample_count, int64_t max_lag)
{
    (void)batch;
    (void)sample_count;
    (void)max_lag;
    return AlignUp(sizeof(ResidualDiagnosticsTilingData), 32U);
}
