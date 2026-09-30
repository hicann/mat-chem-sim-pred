/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "edgepool_sorted_rank_contract_fused_host.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "acl/acl.h"

namespace
{
struct Tiling
{
    uint32_t nodes;
    uint32_t edges;
};
}  // namespace

extern "C" uint32_t aclrtlaunch_edgepool_sorted_rank_contract_fused_kernel(uint32_t block_dim, void* stream,
                                                                           void* order, void* edge_index, void* score,
                                                                           void* cluster, void* selected_edge,
                                                                           void* cluster_score, void* counts,
                                                                           void* workspace, Tiling* tiling);

uint64_t aclnnEdgepoolSortedRankContractFusedGetWorkspaceSize(int64_t nodes, int64_t edges)
{
    if (nodes <= 0 || nodes > INT32_MAX || edges <= 0 || edges > INT32_MAX)
    {
        return UINT64_MAX;
    }
    return 1;
}

int32_t aclnnEdgepoolSortedRankContractFused(void* order, void* edge_index, void* score, void* cluster,
                                             void* selected_edge, void* cluster_score, void* counts, int64_t nodes,
                                             int64_t edges, void* workspace, uint64_t workspace_size, void* stream)
{
    if (order == nullptr || edge_index == nullptr || score == nullptr || cluster == nullptr ||
        selected_edge == nullptr || cluster_score == nullptr || counts == nullptr || stream == nullptr || nodes <= 0 ||
        nodes > INT32_MAX || edges <= 0 || edges > INT32_MAX || workspace_size < 1)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    void* pointers[] = {order, edge_index, score, cluster, selected_edge, cluster_score, counts, workspace};
    for (size_t first = 0; first < 8; ++first)
    {
        for (size_t second = first + 1; second < 8; ++second)
        {
            if (pointers[first] == pointers[second])
            {
                return ACL_ERROR_INVALID_PARAM;
            }
        }
    }
    Tiling tiling{static_cast<uint32_t>(nodes), static_cast<uint32_t>(edges)};
    const uint32_t result = aclrtlaunch_edgepool_sorted_rank_contract_fused_kernel(
        1U, reinterpret_cast<aclrtStream>(stream), order, edge_index, score, cluster, selected_edge, cluster_score,
        counts, workspace, &tiling);
    return result == 0U ? ACL_SUCCESS : static_cast<int32_t>(result);
}
