/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_lightgcn_k2_weighted_sum_fused_host.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../../graph_message_host_common.h"
#include "acl/acl.h"

namespace
{
struct CsrLightgcnK2WeightedSumFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t core_num;
    float alpha0;
    float alpha1;
    float alpha2;
    uint32_t reserved;
};

bool ValidPointers(void* rowPtr, void* sourceIndex, void* norm, void* features, void* output, void* workspace,
                   void* stream)
{
    return !GraphMessageHost::HasNull({rowPtr, sourceIndex, norm, features, output, workspace, stream}) &&
           !GraphMessageHost::Aliases(output, {rowPtr, sourceIndex, norm, features});
}

bool ValidDimensions(int64_t nodes, int64_t edges, int64_t channels)
{
    return GraphMessageHost::PositiveInt32(nodes) && GraphMessageHost::PositiveInt32(edges) &&
           GraphMessageHost::InRange(channels, 512);
}

bool ValidWeights(float alpha0, float alpha1, float alpha2)
{
    return std::isfinite(alpha0) && std::isfinite(alpha1) && std::isfinite(alpha2);
}
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_lightgcn_k2_weighted_sum_stage1_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                           void*, void*, void*, void*, void*);
extern "C" uint32_t aclrtlaunch_csr_lightgcn_k2_weighted_sum_stage2_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                           void*, void*, void*, void*, void*);

extern "C" uint64_t aclnnCsrLightgcnK2WeightedSumFusedGetWorkspaceSize(int64_t nodes, int64_t, int64_t channels)
{
    if (nodes <= 0 || channels <= 0)
    {
        return 0U;
    }
    return GraphMessageHost::AlignUp32(sizeof(CsrLightgcnK2WeightedSumFusedTiling)) +
           GraphMessageHost::AlignUp32(static_cast<uint64_t>(nodes) * channels * sizeof(float));
}
extern "C" int32_t aclnnCsrLightgcnK2WeightedSumFused(void* rowPtr, void* sourceIndex, void* norm, void* features,
                                                      void* output, int64_t nodes, int64_t edges, int64_t channels,
                                                      float alpha0, float alpha1, float alpha2, void* workspace,
                                                      uint64_t workspaceSize, void* stream)
{
    const uint64_t required = aclnnCsrLightgcnK2WeightedSumFusedGetWorkspaceSize(nodes, edges, channels);
    if (!ValidPointers(rowPtr, sourceIndex, norm, features, output, workspace, stream) ||
        !ValidDimensions(nodes, edges, channels) || !ValidWeights(alpha0, alpha1, alpha2) || required == 0U ||
        workspaceSize < required)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint64_t tilingBytes = GraphMessageHost::AlignUp32(sizeof(CsrLightgcnK2WeightedSumFusedTiling));
    auto* temporary = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(workspace) + tilingBytes);
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    CsrLightgcnK2WeightedSumFusedTiling tiling{static_cast<uint32_t>(nodes),
                                               static_cast<uint32_t>(edges),
                                               static_cast<uint32_t>(channels),
                                               cores,
                                               alpha0,
                                               alpha1,
                                               alpha2,
                                               0U};
    const int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    uint32_t result = aclrtlaunch_csr_lightgcn_k2_weighted_sum_stage1_kernel(
        cores, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, norm, features, output, temporary, workspace,
        workspace);
    if (result == 0U)
    {
        result = aclrtlaunch_csr_lightgcn_k2_weighted_sum_stage2_kernel(cores, reinterpret_cast<aclrtStream>(stream),
                                                                        rowPtr, sourceIndex, norm, features, output,
                                                                        temporary, workspace, workspace);
    }
    return static_cast<int32_t>(result);
}
