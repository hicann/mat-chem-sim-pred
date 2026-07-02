/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pid_process_capability_metrics_host.h"

#include <algorithm>

#include "acl/acl.h"

namespace {

struct ProcessCapabilityTilingData {
    uint32_t batch;
    uint32_t sample_count;
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

}  // namespace

extern "C" void aclrtlaunch_pid_process_capability_metrics_kernel(
    uint32_t blockDim, aclrtStream stream, void* values, void* lsl, void* usl, void* metrics, void* tiling);

extern "C" int32_t aclnnPidProcessCapabilityMetrics(
    void* values, void* lsl, void* usl, void* metrics, int64_t batch, int64_t sample_count, void* workspace,
    uint64_t workspace_size, void* stream)
{
    if (values == nullptr || lsl == nullptr || usl == nullptr || metrics == nullptr || workspace == nullptr ||
        stream == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (batch <= 0 || sample_count <= 0 ||
        workspace_size < aclnnPidProcessCapabilityMetricsGetWorkspaceSize(batch, sample_count)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    ProcessCapabilityTilingData tiling;
    tiling.batch = static_cast<uint32_t>(batch);
    tiling.sample_count = static_cast<uint32_t>(sample_count);
    tiling.core_num = ComputeCoreNum(batch);
    tiling.metric_count = static_cast<uint32_t>(PID_PROCESS_CAPABILITY_METRIC_COUNT);

    const auto ret = aclrtMemcpyAsync(
        workspace, sizeof(ProcessCapabilityTilingData), &tiling, sizeof(ProcessCapabilityTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE, reinterpret_cast<aclrtStream>(stream));
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    aclrtlaunch_pid_process_capability_metrics_kernel(
        tiling.core_num, reinterpret_cast<aclrtStream>(stream), values, lsl, usl, metrics, workspace);
    return ACL_SUCCESS;
}

extern "C" uint64_t aclnnPidProcessCapabilityMetricsGetWorkspaceSize(int64_t batch, int64_t sample_count)
{
    (void)batch;
    (void)sample_count;
    return AlignUp(sizeof(ProcessCapabilityTilingData), 32U);
}
