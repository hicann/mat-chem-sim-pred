/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_gatv2_dynamic_attention_aggregate_fused_host.h"

#include "../../attention_host_common.h"

extern "C" uint32_t aclrtlaunch_csr_gatv2_dynamic_attention_aggregate_fused_kernel(uint32_t, aclrtStream, void*, void*,
                                                                                   void*, void*, void*, void*, void*,
                                                                                   void*);

extern "C" uint64_t aclnnCsrGatv2DynamicAttentionAggregateFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t,
                                                                                int64_t)
{
    return AttentionHost::WorkspaceSize();
}

extern "C" int32_t aclnnCsrGatv2DynamicAttentionAggregateFused(void* rowPtr, void* sourceIndex, void* sourceProjected,
                                                               void* targetProjected, void* attention, void* output,
                                                               int64_t nodes, int64_t edges, int64_t heads,
                                                               int64_t channels, int64_t maxSegmentSize,
                                                               float negativeSlope, void* workspace,
                                                               uint64_t workspaceSize, void* stream)
{
    if (!AttentionHost::ValidInvocation(
            {rowPtr, sourceIndex, sourceProjected, targetProjected, attention, output, workspace, stream}, output,
            {rowPtr, sourceIndex, sourceProjected, targetProjected, attention}, nodes, edges, heads, channels,
            maxSegmentSize, workspaceSize))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (!AttentionHost::ValidSlope(negativeSlope))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const AttentionFusedTiling tiling =
        AttentionHost::MakeTiling(nodes, edges, heads, channels, maxSegmentSize, negativeSlope);
    const int32_t copyStatus = AttentionHost::CopyTiling(workspace, workspaceSize, tiling);
    if (copyStatus != ACL_SUCCESS)
    {
        return copyStatus;
    }
    const uint32_t launchStatus = aclrtlaunch_csr_gatv2_dynamic_attention_aggregate_fused_kernel(
        tiling.coreNum, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, sourceProjected, targetProjected,
        attention, output, workspace, workspace);
    return static_cast<int32_t>(launchStatus);
}
