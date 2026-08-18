/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_gat_attention_aggregate_fused_host.h"

#include "../../attention_host_common.h"

extern "C" uint32_t aclrtlaunch_csr_gat_attention_aggregate_fused_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                         void*, void*, void*, void*, void*);

extern "C" uint64_t aclnnCsrGatAttentionAggregateFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t, int64_t)
{
    return AttentionHost::WorkspaceSize();
}

extern "C" int32_t aclnnCsrGatAttentionAggregateFused(void* rowPtr, void* sourceIndex, void* projected,
                                                      void* attentionSource, void* attentionTarget, void* output,
                                                      int64_t nodes, int64_t edges, int64_t heads, int64_t channels,
                                                      int64_t maxSegmentSize, float negativeSlope, void* workspace,
                                                      uint64_t workspaceSize, void* stream)
{
    const bool valid = AttentionHost::ValidInvocation(
        {rowPtr, sourceIndex, projected, attentionSource, attentionTarget, output, workspace, stream}, output,
        {rowPtr, sourceIndex, projected, attentionSource, attentionTarget}, nodes, edges, heads, channels,
        maxSegmentSize, workspaceSize);
    if (!valid || !AttentionHost::ValidSlope(negativeSlope))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const AttentionFusedTiling tiling =
        AttentionHost::MakeTiling(nodes, edges, heads, channels, maxSegmentSize, negativeSlope);
    const int32_t copied = AttentionHost::CopyTiling(workspace, workspaceSize, tiling);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    return static_cast<int32_t>(aclrtlaunch_csr_gat_attention_aggregate_fused_kernel(
        tiling.coreNum, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, projected, attentionSource,
        attentionTarget, output, workspace, workspace));
}
