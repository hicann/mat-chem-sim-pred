/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pid_tuning_rule_batch_host.h"

#include <algorithm>
#include <limits>

#include "acl/acl.h"

namespace {

struct TuningRuleBatchTilingData {
    uint32_t batch;
    uint32_t core_num;
    uint32_t rule_count;
    uint32_t param_count;
    uint32_t diagnostic_count;
};

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return ((value + align - 1U) / align) * align;
}

uint32_t ComputeCoreNum(int64_t batch)
{
    // This kernel writes its per-item outputs to GM with scalar SetValue. On
    // Ascend910B the data cache holding those scalar writes is not reliably
    // flushed to HBM across a multi-core launch (an item whose record shares a
    // 64B cache line with the next core's record can be read back stale), so
    // results are non-deterministic when more than one AI core is used. Pinning
    // to a single core makes the GM writeback deterministic; this is cheap
    // post-processing, so the loss of multi-core parallelism is negligible.
    (void)batch;
    return 1U;
}

bool HasValidShape(int64_t batch)
{
    constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    return batch > 0 && batch <= kUint32Max;
}

}  // namespace

extern "C" void aclrtlaunch_pid_tuning_rule_batch_kernel(
    uint32_t blockDim, aclrtStream stream, void* process_gain, void* time_constant, void* dead_time,
    void* lambda_value, void* pid_params, void* diagnostics, void* tiling);

extern "C" int32_t aclnnPidTuningRuleBatch(
    void* process_gain, void* time_constant, void* dead_time, void* lambda_value, void* pid_params,
    void* diagnostics, int64_t batch, void* workspace, uint64_t workspace_size, void* stream)
{
    if (process_gain == nullptr || time_constant == nullptr || dead_time == nullptr || lambda_value == nullptr ||
        pid_params == nullptr || diagnostics == nullptr || workspace == nullptr || stream == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (!HasValidShape(batch) || workspace_size < aclnnPidTuningRuleBatchGetWorkspaceSize(batch)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    TuningRuleBatchTilingData tiling;
    tiling.batch = static_cast<uint32_t>(batch);
    tiling.core_num = ComputeCoreNum(batch);
    tiling.rule_count = static_cast<uint32_t>(PID_TUNING_RULE_BATCH_RULE_COUNT);
    tiling.param_count = static_cast<uint32_t>(PID_TUNING_RULE_BATCH_PARAM_COUNT);
    tiling.diagnostic_count = static_cast<uint32_t>(PID_TUNING_RULE_BATCH_DIAGNOSTIC_COUNT);

    const auto ret = aclrtMemcpyAsync(
        workspace, sizeof(TuningRuleBatchTilingData), &tiling, sizeof(TuningRuleBatchTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE, reinterpret_cast<aclrtStream>(stream));
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    aclrtlaunch_pid_tuning_rule_batch_kernel(
        tiling.core_num, reinterpret_cast<aclrtStream>(stream), process_gain, time_constant, dead_time, lambda_value,
        pid_params, diagnostics, workspace);
    return ACL_SUCCESS;
}

extern "C" uint64_t aclnnPidTuningRuleBatchGetWorkspaceSize(int64_t batch)
{
    (void)batch;
    return AlignUp(sizeof(TuningRuleBatchTilingData), 32U);
}
