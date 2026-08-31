/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_point_transformer_attention_aggregate_fused_host.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>

#include "acl/acl.h"

namespace
{
struct CsrPointTransformerAttentionAggregateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t max_segment_size;
    uint32_t core_num;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};

struct Invocation
{
    void* row_ptr;
    void* source_index;
    void* alpha_source;
    void* alpha_target;
    void* value;
    void* delta;
    void* output;
    int64_t nodes;
    int64_t edges;
    int64_t channels;
    int64_t max_segment_size;
    int64_t dtype;
    void* workspace;
    uint64_t workspace_size;
    void* stream;
};

uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }
bool EqualsAny(void* pointer, std::initializer_list<void*> values)
{
    return std::find(values.begin(), values.end(), pointer) != values.end();
}

bool HasValidPointers(const Invocation& invocation)
{
    return invocation.row_ptr != nullptr && invocation.source_index != nullptr && invocation.alpha_source != nullptr &&
           invocation.alpha_target != nullptr && invocation.value != nullptr && invocation.delta != nullptr &&
           invocation.output != nullptr && invocation.workspace != nullptr && invocation.stream != nullptr;
}

bool HasIndependentOutputs(const Invocation& invocation)
{
    return !EqualsAny(invocation.output,
                      {invocation.row_ptr, invocation.source_index, invocation.alpha_source, invocation.alpha_target,
                       invocation.value, invocation.delta, invocation.workspace}) &&
           !EqualsAny(invocation.workspace,
                      {invocation.row_ptr, invocation.source_index, invocation.alpha_source, invocation.alpha_target,
                       invocation.value, invocation.delta, invocation.output});
}

bool HasValidShape(const Invocation& invocation, uint64_t required)
{
    return invocation.workspace_size >= required && invocation.nodes > 0 && invocation.nodes <= INT32_MAX &&
           invocation.edges > 0 && invocation.edges <= INT32_MAX && invocation.channels > 0 &&
           invocation.channels <= 128 && invocation.max_segment_size > 0 && invocation.max_segment_size <= 512 &&
           invocation.dtype >= 0 && invocation.dtype <= 2;
}
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_point_transformer_attention_aggregate_fused_kernel_fp32(uint32_t, aclrtStream,
                                                                                            void*, void*, void*, void*,
                                                                                            void*, void*, void*, void*,
                                                                                            void*);
extern "C" uint32_t aclrtlaunch_csr_point_transformer_attention_aggregate_fused_kernel_fp16(uint32_t, aclrtStream,
                                                                                            void*, void*, void*, void*,
                                                                                            void*, void*, void*, void*,
                                                                                            void*);
extern "C" uint32_t aclrtlaunch_csr_point_transformer_attention_aggregate_fused_kernel_bf16(uint32_t, aclrtStream,
                                                                                            void*, void*, void*, void*,
                                                                                            void*, void*, void*, void*,
                                                                                            void*);

extern "C" uint64_t aclnnCsrPointTransformerAttentionAggregateFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t)
{
    return AlignUp(sizeof(CsrPointTransformerAttentionAggregateFusedTiling));
}

extern "C" int32_t aclnnCsrPointTransformerAttentionAggregateFused(void* rowPtr, void* sourceIndex, void* alphaSource,
                                                                   void* alphaTarget, void* value, void* delta,
                                                                   void* output, int64_t nodes, int64_t edges,
                                                                   int64_t channels, int64_t maxSegmentSize,
                                                                   int64_t dtype, void* workspace,
                                                                   uint64_t workspaceSize, void* stream)
{
    const Invocation invocation{rowPtr,       sourceIndex, alphaSource, alphaTarget, value,
                                delta,        output,      nodes,       edges,       channels,
                                maxSegmentSize, dtype,     workspace,   workspaceSize, stream};
    const uint64_t required =
        aclnnCsrPointTransformerAttentionAggregateFusedGetWorkspaceSize(nodes, edges, channels, maxSegmentSize);
    if (!HasValidPointers(invocation) || !HasIndependentOutputs(invocation) || !HasValidShape(invocation, required))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    CsrPointTransformerAttentionAggregateFusedTiling tiling{static_cast<uint32_t>(nodes),
                                                            static_cast<uint32_t>(edges),
                                                            static_cast<uint32_t>(channels),
                                                            static_cast<uint32_t>(maxSegmentSize),
                                                            cores,
                                                            0U,
                                                            0U,
                                                            0U};
    const int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    const aclrtStream aclStream = reinterpret_cast<aclrtStream>(stream);
    uint32_t result = 0U;
    if (dtype == 1)
    {
        result = aclrtlaunch_csr_point_transformer_attention_aggregate_fused_kernel_fp16(
            cores, aclStream, rowPtr, sourceIndex, alphaSource, alphaTarget, value, delta, output, workspace,
            workspace);
    }
    else if (dtype == 2)
    {
        result = aclrtlaunch_csr_point_transformer_attention_aggregate_fused_kernel_bf16(
            cores, aclStream, rowPtr, sourceIndex, alphaSource, alphaTarget, value, delta, output, workspace,
            workspace);
    }
    else
    {
        result = aclrtlaunch_csr_point_transformer_attention_aggregate_fused_kernel_fp32(
            cores, aclStream, rowPtr, sourceIndex, alphaSource, alphaTarget, value, delta, output, workspace,
            workspace);
    }
    return static_cast<int32_t>(result);
}
