/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "ppf_point_pair_features_fused_host.h"

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"

extern "C" uint32_t aclrtlaunch_ppf_point_pair_features_fused_kernel(uint32_t, aclrtStream, void*, void*, void*, void*,
                                                                     void*, uint32_t, uint32_t);

extern "C" uint64_t aclnnPpfPointPairFeaturesFusedGetWorkspaceSize(int64_t nodes, int64_t edges)
{
    (void)nodes;
    (void)edges;
    return 0U;
}

extern "C" int32_t aclnnPpfPointPairFeaturesFused(void* position, void* normal, void* sourceIndex, void* targetIndex,
                                                  void* output, int64_t nodes, int64_t edges, void* workspace,
                                                  uint64_t workspaceSize, void* stream)
{
    if (position == nullptr || normal == nullptr || sourceIndex == nullptr || targetIndex == nullptr ||
        output == nullptr || stream == nullptr || workspaceSize != 0U || nodes <= 0 || nodes > INT32_MAX ||
        edges <= 0 || edges > INT32_MAX || position == normal || position == output || normal == output ||
        sourceIndex == targetIndex)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    (void)workspace;
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(edges, 40)));
    return static_cast<int32_t>(aclrtlaunch_ppf_point_pair_features_fused_kernel(
        cores, reinterpret_cast<aclrtStream>(stream), position, normal, sourceIndex, targetIndex, output,
        static_cast<uint32_t>(nodes), static_cast<uint32_t>(edges)));
}
