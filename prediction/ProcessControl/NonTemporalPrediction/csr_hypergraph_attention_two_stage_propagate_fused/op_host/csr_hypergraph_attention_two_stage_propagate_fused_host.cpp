/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_hypergraph_attention_two_stage_propagate_fused_host.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "acl/acl.h"

namespace
{
struct CsrHypergraphAttentionTwoStagePropagateFusedTiling
{
    uint32_t nodes;
    uint32_t hyperedges;
    uint32_t incidences;
    uint32_t heads;
    uint32_t channels;
    uint32_t max_edge_size;
    uint32_t max_node_degree;
    float negative_slope;
};
uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }
bool TensorBytes(int64_t rows, int64_t columns, uint64_t& bytes)
{
    if (rows <= 0 || columns <= 0 ||
        static_cast<uint64_t>(rows) >
            std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(columns) / sizeof(float))
    {
        return false;
    }
    bytes = static_cast<uint64_t>(rows) * columns * sizeof(float);
    return true;
}
bool EqualsAny(void* pointer, std::initializer_list<void*> values)
{
    return std::find(values.begin(), values.end(), pointer) != values.end();
}
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_hypergraph_attention_node_to_edge_kernel_fp32(uint32_t, aclrtStream, void*, void*,
                                                                                  void*, void*, void*, void*, void*,
                                                                                  void*, void*);
extern "C" uint32_t aclrtlaunch_csr_hypergraph_attention_edge_to_node_kernel_fp32(uint32_t, aclrtStream, void*, void*,
                                                                                  void*, void*, void*, void*, void*,
                                                                                  void*, void*);
extern "C" uint32_t aclrtlaunch_csr_hypergraph_attention_node_to_edge_kernel_fp16(uint32_t, aclrtStream, void*, void*,
                                                                                  void*, void*, void*, void*, void*,
                                                                                  void*, void*);
extern "C" uint32_t aclrtlaunch_csr_hypergraph_attention_edge_to_node_kernel_fp16(uint32_t, aclrtStream, void*, void*,
                                                                                  void*, void*, void*, void*, void*,
                                                                                  void*, void*);
extern "C" uint32_t aclrtlaunch_csr_hypergraph_attention_node_to_edge_kernel_bf16(uint32_t, aclrtStream, void*, void*,
                                                                                  void*, void*, void*, void*, void*,
                                                                                  void*, void*);
extern "C" uint32_t aclrtlaunch_csr_hypergraph_attention_edge_to_node_kernel_bf16(uint32_t, aclrtStream, void*, void*,
                                                                                  void*, void*, void*, void*, void*,
                                                                                  void*, void*);

namespace
{
struct Pointers
{
    void* edgeRowPtr;
    void* nodeIndex;
    void* edgeScale;
    void* nodeRowPtr;
    void* edgeIndex;
    void* incidencePosition;
    void* nodeScale;
    void* features;
    void* attentionLogits;
    void* output;
    void* workspace;
    void* stream;
};

struct Shape
{
    int64_t nodes;
    int64_t hyperedges;
    int64_t incidences;
    int64_t heads;
    int64_t channels;
    int64_t maxEdgeSize;
    int64_t maxNodeDegree;
    int64_t dtype;
    float negativeSlope;
};

bool InvalidPointers(const Pointers& value)
{
    const std::initializer_list<void*> inputs{value.edgeRowPtr, value.nodeIndex, value.edgeScale,
                                              value.nodeRowPtr, value.edgeIndex, value.incidencePosition,
                                              value.nodeScale,  value.features,  value.attentionLogits};
    return EqualsAny(nullptr, inputs) || value.output == nullptr || value.workspace == nullptr ||
           value.stream == nullptr || EqualsAny(value.output, inputs) || EqualsAny(value.workspace, inputs) ||
           value.output == value.workspace;
}

bool InvalidShape(const Shape& value)
{
    return value.nodes <= 0 || value.nodes > INT32_MAX || value.hyperedges <= 0 || value.hyperedges > INT32_MAX ||
           value.incidences <= 0 || value.incidences > INT32_MAX || value.heads <= 0 || value.heads > 4 ||
           value.channels <= 0 || value.channels > 32 || value.maxEdgeSize <= 0 || value.maxEdgeSize > 512 ||
           value.maxNodeDegree <= 0 || value.maxNodeDegree > 512 || value.dtype < 0 || value.dtype > 2 ||
           !std::isfinite(value.negativeSlope) || value.negativeSlope < 0.0F || value.negativeSlope > 1.0F;
}

using KernelLaunch = uint32_t (*)(uint32_t, aclrtStream, void*, void*, void*, void*, void*, void*, void*, void*, void*);

struct Launchers
{
    KernelLaunch nodeToEdge;
    KernelLaunch edgeToNode;
};

Launchers SelectLaunchers(int64_t dtype)
{
    if (dtype == 1)
    {
        return {aclrtlaunch_csr_hypergraph_attention_node_to_edge_kernel_fp16,
                aclrtlaunch_csr_hypergraph_attention_edge_to_node_kernel_fp16};
    }
    if (dtype == 2)
    {
        return {aclrtlaunch_csr_hypergraph_attention_node_to_edge_kernel_bf16,
                aclrtlaunch_csr_hypergraph_attention_edge_to_node_kernel_bf16};
    }
    return {aclrtlaunch_csr_hypergraph_attention_node_to_edge_kernel_fp32,
            aclrtlaunch_csr_hypergraph_attention_edge_to_node_kernel_fp32};
}

int32_t LaunchKernels(const Pointers& pointers, const Shape& shape, void* edgeFeatures, void* normalizedAttention)
{
    const uint32_t edgeCores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(shape.hyperedges, 40)));
    const uint32_t nodeCores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(shape.nodes, 40)));
    const aclrtStream stream = reinterpret_cast<aclrtStream>(pointers.stream);
    const Launchers launchers = SelectLaunchers(shape.dtype);
    uint32_t result = launchers.nodeToEdge(edgeCores, stream, pointers.edgeRowPtr, pointers.nodeIndex,
                                           pointers.edgeScale, pointers.features, pointers.attentionLogits,
                                           normalizedAttention, edgeFeatures, pointers.workspace, pointers.workspace);
    if (result == 0U)
    {
        result = launchers.edgeToNode(nodeCores, stream, pointers.nodeRowPtr, pointers.edgeIndex,
                                      pointers.incidencePosition, pointers.nodeScale, normalizedAttention, edgeFeatures,
                                      pointers.output, pointers.workspace, pointers.workspace);
    }
    return static_cast<int32_t>(result);
}
}  // namespace

extern "C" uint64_t aclnnCsrHypergraphAttentionTwoStagePropagateFusedGetWorkspaceSize(int64_t, int64_t hyperedges,
                                                                                      int64_t incidences, int64_t heads,
                                                                                      int64_t channels, int64_t,
                                                                                      int64_t)
{
    uint64_t featureBytes = 0U, attentionBytes = 0U;
    if (!TensorBytes(hyperedges, heads * channels, featureBytes) || !TensorBytes(incidences, heads, attentionBytes))
    {
        return 0U;
    }
    return AlignUp(sizeof(CsrHypergraphAttentionTwoStagePropagateFusedTiling)) + AlignUp(featureBytes) +
           AlignUp(attentionBytes);
}

extern "C" int32_t aclnnCsrHypergraphAttentionTwoStagePropagateFused(
    void* edgeRowPtr, void* nodeIndex, void* edgeScale, void* nodeRowPtr, void* edgeIndex, void* incidencePosition,
    void* nodeScale, void* features, void* attentionLogits, void* output, int64_t nodes, int64_t hyperedges,
    int64_t incidences, int64_t heads, int64_t channels, int64_t maxEdgeSize, int64_t maxNodeDegree, int64_t dtype,
    float negativeSlope, void* workspace, uint64_t workspaceSize, void* stream)
{
    const Pointers pointers{edgeRowPtr, nodeIndex, edgeScale,       nodeRowPtr, edgeIndex, incidencePosition,
                            nodeScale,  features,  attentionLogits, output,     workspace, stream};
    const Shape shape{nodes, hyperedges, incidences, heads, channels, maxEdgeSize, maxNodeDegree, dtype, negativeSlope};
    const uint64_t required = aclnnCsrHypergraphAttentionTwoStagePropagateFusedGetWorkspaceSize(
        nodes, hyperedges, incidences, heads, channels, maxEdgeSize, maxNodeDegree);
    if (InvalidPointers(pointers) || InvalidShape(shape) || required == 0U || workspaceSize < required)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    CsrHypergraphAttentionTwoStagePropagateFusedTiling tiling{
        static_cast<uint32_t>(nodes),         static_cast<uint32_t>(hyperedges),
        static_cast<uint32_t>(incidences),    static_cast<uint32_t>(heads),
        static_cast<uint32_t>(channels),      static_cast<uint32_t>(maxEdgeSize),
        static_cast<uint32_t>(maxNodeDegree), negativeSlope};
    const int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    uint64_t edgeFeatureBytes = 0U;
    (void)TensorBytes(hyperedges, heads * channels, edgeFeatureBytes);
    auto* edgeFeatures = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(workspace) + AlignUp(sizeof(tiling)));
    auto* normalizedAttention =
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(edgeFeatures) + AlignUp(edgeFeatureBytes));
    return LaunchKernels(pointers, shape, edgeFeatures, normalizedAttention);
}
