/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_transformer_dot_attention_aggregate_fused_host.h"

#include <cmath>

#include "../../attention_host_common.h"

extern "C" uint32_t aclrtlaunch_csr_transformer_dot_attention_aggregate_fused_kernel(uint32_t, aclrtStream, void*,
                                                                                     void*, void*, void*, void*, void*,
                                                                                     void*, void*);

extern "C" uint64_t aclnnCsrTransformerDotAttentionAggregateFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t,
                                                                                  int64_t)
{
    return AttentionHost::WorkspaceSize();
}

extern "C" int32_t aclnnCsrTransformerDotAttentionAggregateFused(void* rowPtr, void* sourceIndex, void* query,
                                                                 void* key, void* value, void* output, int64_t nodes,
                                                                 int64_t edges, int64_t heads, int64_t channels,
                                                                 int64_t maxSegmentSize, void* workspace,
                                                                 uint64_t workspaceSize, void* stream)
{
    const bool invocationOk = AttentionHost::ValidInvocation(
        {rowPtr, sourceIndex, query, key, value, output, workspace, stream}, output,
        {rowPtr, sourceIndex, query, key, value}, nodes, edges, heads, channels, maxSegmentSize, workspaceSize);
    if (!invocationOk)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const float inverseScale = 1.0F / std::sqrt(static_cast<float>(channels));
    const AttentionFusedTiling tiling =
        AttentionHost::MakeTiling(nodes, edges, heads, channels, maxSegmentSize, inverseScale);
    const int32_t copyResult = AttentionHost::CopyTiling(workspace, workspaceSize, tiling);
    if (copyResult != ACL_SUCCESS)
    {
        return copyResult;
    }
    return static_cast<int32_t>(aclrtlaunch_csr_transformer_dot_attention_aggregate_fused_kernel(
        tiling.coreNum, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, query, key, value, output,
        workspace, workspace));
}
