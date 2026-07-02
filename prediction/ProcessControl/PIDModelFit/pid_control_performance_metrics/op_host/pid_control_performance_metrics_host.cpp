/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pid_control_performance_metrics_host.h"

#include <algorithm>

#include "acl/acl.h"

namespace {

struct ControlPerformanceTilingData {
    uint32_t batch;
    uint32_t sample_count;
    uint32_t core_num;
    uint32_t metric_count;
    float sample_interval;
    float settle_band;
};

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return ((value + align - 1U) / align) * align;
}

uint32_t ComputeCoreNum(int64_t batch)
{
    // This kernel writes its per-batch metrics to GM with scalar SetValue. On
    // Ascend910B the data cache holding those scalar writes is not reliably
    // flushed to HBM across a multi-core launch (a batch whose 80-byte metrics
    // record straddles a 64B cache line shared with the next core's record can
    // be read back stale), so results are non-deterministic when more than one
    // AI core is used. Pinning to a single core makes the GM writeback
    // deterministic; the metric computation is tiny post-processing relative to
    // the rest of the tuning pipeline, so the loss of multi-core parallelism is
    // negligible here.
    (void)batch;
    return 1U;
}

}  // namespace

extern "C" void aclrtlaunch_pid_control_performance_metrics_kernel(
    uint32_t blockDim, aclrtStream stream, void* pv, void* sp, void* lsl, void* usl, void* mv_variance,
    void* metrics, void* tiling);

extern "C" int32_t aclnnPidControlPerformanceMetrics(
    void* pv, void* sp, void* lsl, void* usl, void* mv_variance, void* metrics, int64_t batch, int64_t sample_count,
    float sample_interval, float settle_band, void* workspace, uint64_t workspace_size, void* stream)
{
    if (pv == nullptr || sp == nullptr || lsl == nullptr || usl == nullptr || mv_variance == nullptr ||
        metrics == nullptr || workspace == nullptr || stream == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (batch <= 0 || sample_count <= 0 ||
        workspace_size < aclnnPidControlPerformanceMetricsGetWorkspaceSize(batch, sample_count)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    ControlPerformanceTilingData tiling;
    tiling.batch = static_cast<uint32_t>(batch);
    tiling.sample_count = static_cast<uint32_t>(sample_count);
    tiling.core_num = ComputeCoreNum(batch);
    tiling.metric_count = static_cast<uint32_t>(PID_CONTROL_PERFORMANCE_METRIC_COUNT);
    tiling.sample_interval = sample_interval;
    tiling.settle_band = settle_band;

    const auto ret = aclrtMemcpyAsync(
        workspace, sizeof(ControlPerformanceTilingData), &tiling, sizeof(ControlPerformanceTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE, reinterpret_cast<aclrtStream>(stream));
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    aclrtlaunch_pid_control_performance_metrics_kernel(
        tiling.core_num, reinterpret_cast<aclrtStream>(stream), pv, sp, lsl, usl, mv_variance, metrics, workspace);
    return ACL_SUCCESS;
}

extern "C" uint64_t aclnnPidControlPerformanceMetricsGetWorkspaceSize(int64_t batch, int64_t sample_count)
{
    (void)batch;
    (void)sample_count;
    return AlignUp(sizeof(ControlPerformanceTilingData), 32U);
}
