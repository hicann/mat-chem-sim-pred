/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_agnn_cosine_attention_aggregate_fused_host.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "acl/acl.h"

namespace
{
struct CsrAgnnCosineAttentionAggregateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t core_num;
    uint32_t max_segment_size;
    float beta;
    uint32_t reserved0;
    uint32_t reserved1;
};

uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }
bool ValidDimensions(int64_t nodes, int64_t edges, int64_t channels, int64_t maxSegmentSize, float beta)
{
    return nodes > 0 && nodes <= INT32_MAX && edges > 0 && edges <= INT32_MAX && channels > 0 && channels <= 512 &&
           maxSegmentSize > 0 && maxSegmentSize <= 2048 && std::isfinite(beta);
}
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_agnn_cosine_attention_aggregate_fused_kernel(uint32_t, aclrtStream, void*, void*,
                                                                                 void*, void*, void*, void*, void*);

extern "C" uint64_t aclnnCsrAgnnCosineAttentionAggregateFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t)
{
    return AlignUp(sizeof(CsrAgnnCosineAttentionAggregateFusedTiling));
}

extern "C" int32_t aclnnCsrAgnnCosineAttentionAggregateFused(void* rowPtr, void* sourceIndex, void* features,
                                                             void* normalized, void* output, int64_t nodes,
                                                             int64_t edges, int64_t channels, int64_t maxSegmentSize,
                                                             float beta, void* workspace, uint64_t workspaceSize,
                                                             void* stream)
{
    const uint64_t required =
        aclnnCsrAgnnCosineAttentionAggregateFusedGetWorkspaceSize(nodes, edges, channels, maxSegmentSize);
    if (rowPtr == nullptr || sourceIndex == nullptr || features == nullptr || normalized == nullptr ||
        output == nullptr || workspace == nullptr || stream == nullptr || output == rowPtr || output == sourceIndex ||
        output == features || output == normalized || workspaceSize < required ||
        !ValidDimensions(nodes, edges, channels, maxSegmentSize, beta))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    CsrAgnnCosineAttentionAggregateFusedTiling tiling{static_cast<uint32_t>(nodes),
                                                      static_cast<uint32_t>(edges),
                                                      static_cast<uint32_t>(channels),
                                                      cores,
                                                      static_cast<uint32_t>(maxSegmentSize),
                                                      beta,
                                                      0U,
                                                      0U};
    const int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    return static_cast<int32_t>(aclrtlaunch_csr_agnn_cosine_attention_aggregate_fused_kernel(
        cores, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, features, normalized, output, workspace,
        workspace));
}
