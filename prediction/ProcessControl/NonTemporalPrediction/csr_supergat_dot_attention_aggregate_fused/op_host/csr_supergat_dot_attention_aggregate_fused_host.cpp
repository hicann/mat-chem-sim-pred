/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_supergat_dot_attention_aggregate_fused_host.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>

#include "acl/acl.h"

namespace
{
struct CsrSuperGatDotAttentionAggregateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t heads;
    uint32_t channels;
    uint32_t max_segment_size;
    uint32_t core_num;
    float inverse_scale;
    float negative_slope;
};

uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }
bool EqualsAny(void* pointer, std::initializer_list<void*> values)
{
    return std::find(values.begin(), values.end(), pointer) != values.end();
}
bool ValidRequest(void* rowPtr, void* sourceIndex, void* projected, void* output, void* workspace, void* stream,
                  uint64_t workspaceSize, uint64_t required, int64_t nodes, int64_t edges, int64_t heads,
                  int64_t channels, int64_t maxSegmentSize, int64_t dtype, float negativeSlope)
{
    const bool null_arg = rowPtr == nullptr || sourceIndex == nullptr || projected == nullptr || output == nullptr ||
                          workspace == nullptr || stream == nullptr;
    const bool alias = EqualsAny(output, {rowPtr, sourceIndex, projected, workspace}) ||
                       EqualsAny(workspace, {rowPtr, sourceIndex, projected, output});
    const bool shape = nodes <= 0 || nodes > INT32_MAX || edges <= 0 || edges > INT32_MAX || heads <= 0 || heads > 8 ||
                       channels <= 0 || channels > 64 || maxSegmentSize <= 0 || maxSegmentSize > 512;
    return !null_arg && !alias && workspaceSize >= required && !shape && dtype >= 0 && dtype <= 2 &&
           std::isfinite(negativeSlope) && negativeSlope >= 0.0F && negativeSlope <= 1.0F;
}
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_supergat_dot_attention_aggregate_fused_kernel_fp32(uint32_t, aclrtStream, void*,
                                                                                       void*, void*, void*, void*,
                                                                                       void*);
extern "C" uint32_t aclrtlaunch_csr_supergat_dot_attention_aggregate_fused_kernel_fp16(uint32_t, aclrtStream, void*,
                                                                                       void*, void*, void*, void*,
                                                                                       void*);
extern "C" uint32_t aclrtlaunch_csr_supergat_dot_attention_aggregate_fused_kernel_bf16(uint32_t, aclrtStream, void*,
                                                                                       void*, void*, void*, void*,
                                                                                       void*);

extern "C" uint64_t aclnnCsrSuperGatDotAttentionAggregateFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t,
                                                                               int64_t)
{
    return AlignUp(sizeof(CsrSuperGatDotAttentionAggregateFusedTiling));
}

extern "C" int32_t aclnnCsrSuperGatDotAttentionAggregateFused(void* rowPtr, void* sourceIndex, void* projected,
                                                              void* output, int64_t nodes, int64_t edges, int64_t heads,
                                                              int64_t channels, int64_t maxSegmentSize, int64_t dtype,
                                                              float negativeSlope, void* workspace,
                                                              uint64_t workspaceSize, void* stream)
{
    const uint64_t required =
        aclnnCsrSuperGatDotAttentionAggregateFusedGetWorkspaceSize(nodes, edges, heads, channels, maxSegmentSize);
    if (!ValidRequest(rowPtr, sourceIndex, projected, output, workspace, stream, workspaceSize, required, nodes, edges,
                      heads, channels, maxSegmentSize, dtype, negativeSlope))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    CsrSuperGatDotAttentionAggregateFusedTiling tiling{static_cast<uint32_t>(nodes),
                                                       static_cast<uint32_t>(edges),
                                                       static_cast<uint32_t>(heads),
                                                       static_cast<uint32_t>(channels),
                                                       static_cast<uint32_t>(maxSegmentSize),
                                                       cores,
                                                       1.0F / std::sqrt(static_cast<float>(channels)),
                                                       negativeSlope};
    const int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    const aclrtStream aclStream = reinterpret_cast<aclrtStream>(stream);
    uint32_t result = 0U;
    if (dtype == 1)
    {
        result = aclrtlaunch_csr_supergat_dot_attention_aggregate_fused_kernel_fp16(
            cores, aclStream, rowPtr, sourceIndex, projected, output, workspace, workspace);
    }
    else if (dtype == 2)
    {
        result = aclrtlaunch_csr_supergat_dot_attention_aggregate_fused_kernel_bf16(
            cores, aclStream, rowPtr, sourceIndex, projected, output, workspace, workspace);
    }
    else
    {
        result = aclrtlaunch_csr_supergat_dot_attention_aggregate_fused_kernel_fp32(
            cores, aclStream, rowPtr, sourceIndex, projected, output, workspace, workspace);
    }
    return static_cast<int32_t>(result);
}
