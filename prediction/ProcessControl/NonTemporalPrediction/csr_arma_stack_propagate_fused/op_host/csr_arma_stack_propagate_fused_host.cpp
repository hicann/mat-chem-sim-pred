/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_arma_stack_propagate_fused_host.h"

#include <algorithm>
#include <array>
#include <cstdint>

#include "acl/acl.h"

namespace
{
struct CsrArmaStackPropagateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t stacks;
    uint32_t channels;
    uint32_t core_num;
    uint32_t relu;
    uint32_t reserved[2];
};

uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }

bool HasNull(const std::array<const void*, 9>& pointers)
{
    return std::any_of(pointers.begin(), pointers.end(), [](const void* value) { return value == nullptr; });
}

bool OutputAliases(void* output, const std::array<void*, 6>& inputs)
{
    return std::find(inputs.begin(), inputs.end(), output) != inputs.end();
}

bool InvalidShape(int64_t nodes, int64_t edges, int64_t stacks, int64_t channels)
{
    return nodes <= 0 || nodes > INT32_MAX || edges <= 0 || edges > INT32_MAX || stacks <= 0 || stacks > 64 ||
           channels <= 0 || channels > 1024 || nodes > INT32_MAX / stacks;
}
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_arma_stack_propagate_fused_kernel(uint32_t, aclrtStream, void*, void*, void*, void*,
                                                                      void*, void*, void*, void*, void*);

extern "C" uint64_t aclnnCsrArmaStackPropagateFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t)
{
    return AlignUp(sizeof(CsrArmaStackPropagateFusedTiling));
}

extern "C" int32_t aclnnCsrArmaStackPropagateFused(void* rowPtr, void* sourceIndex, void* edgeWeight, void* projected,
                                                   void* root, void* bias, void* output, int64_t nodes, int64_t edges,
                                                   int64_t stacks, int64_t channels, int64_t relu, void* workspace,
                                                   uint64_t workspaceSize, void* stream)
{
    const uint64_t required = aclnnCsrArmaStackPropagateFusedGetWorkspaceSize(nodes, edges, stacks, channels);
    const std::array<const void*, 9> pointers{rowPtr, sourceIndex, edgeWeight, projected, root,
                                              bias,   output,      workspace,  stream};
    const std::array<void*, 6> inputs{rowPtr, sourceIndex, edgeWeight, projected, root, bias};
    if (HasNull(pointers) || OutputAliases(output, inputs) || workspaceSize < required ||
        InvalidShape(nodes, edges, stacks, channels) || (relu != 0 && relu != 1))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const int64_t workItems = nodes * stacks;
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(workItems, 40)));
    CsrArmaStackPropagateFusedTiling tiling{static_cast<uint32_t>(nodes),
                                            static_cast<uint32_t>(edges),
                                            static_cast<uint32_t>(stacks),
                                            static_cast<uint32_t>(channels),
                                            cores,
                                            static_cast<uint32_t>(relu),
                                            {0U, 0U}};
    const int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    return static_cast<int32_t>(aclrtlaunch_csr_arma_stack_propagate_fused_kernel(
        cores, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, edgeWeight, projected, root, bias, output,
        workspace, workspace));
}
