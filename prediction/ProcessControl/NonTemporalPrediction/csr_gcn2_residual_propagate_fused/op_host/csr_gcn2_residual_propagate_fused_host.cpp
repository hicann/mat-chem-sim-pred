/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_gcn2_residual_propagate_fused_host.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "acl/acl.h"

namespace
{
struct CsrGcn2ResidualPropagateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t core_num;
    float alpha;
    uint32_t reserved[3];
};

uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }

bool HasNull(const std::array<const void*, 8>& pointers)
{
    return std::any_of(pointers.begin(), pointers.end(), [](const void* value) { return value == nullptr; });
}

bool InvalidShape(int64_t nodes, int64_t edges, int64_t channels)
{
    return nodes <= 0 || nodes > INT32_MAX || edges <= 0 || edges > INT32_MAX || channels <= 0 || channels > 1024;
}

bool InvalidAlpha(float alpha) { return !std::isfinite(alpha) || alpha < 0.0F || alpha > 1.0F; }
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_gcn2_residual_propagate_fused_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                         void*, void*, void*, void*, void*);

extern "C" uint64_t aclnnCsrGcn2ResidualPropagateFusedGetWorkspaceSize(int64_t, int64_t, int64_t)
{
    return AlignUp(sizeof(CsrGcn2ResidualPropagateFusedTiling));
}

extern "C" int32_t aclnnCsrGcn2ResidualPropagateFused(void* rowPtr, void* sourceIndex, void* edgeWeight, void* current,
                                                      void* initial, void* output, int64_t nodes, int64_t edges,
                                                      int64_t channels, float alpha, void* workspace,
                                                      uint64_t workspaceSize, void* stream)
{
    const uint64_t required = aclnnCsrGcn2ResidualPropagateFusedGetWorkspaceSize(nodes, edges, channels);
    const std::array<const void*, 8> pointers{rowPtr,  sourceIndex, edgeWeight, current,
                                              initial, output,      workspace,  stream};
    if (HasNull(pointers) || output == current || output == initial || workspaceSize < required ||
        InvalidShape(nodes, edges, channels) || InvalidAlpha(alpha))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    CsrGcn2ResidualPropagateFusedTiling tiling{static_cast<uint32_t>(nodes),
                                               static_cast<uint32_t>(edges),
                                               static_cast<uint32_t>(channels),
                                               cores,
                                               alpha,
                                               {0U, 0U, 0U}};
    const int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    return static_cast<int32_t>(aclrtlaunch_csr_gcn2_residual_propagate_fused_kernel(
        cores, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, edgeWeight, current, initial, output,
        workspace, workspace));
}
