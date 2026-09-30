/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <cstdint>

extern "C"
{
    uint64_t aclnnEdgepoolSortedRankContractFusedGetWorkspaceSize(int64_t nodes, int64_t edges);

    int32_t aclnnEdgepoolSortedRankContractFused(void* order, void* edge_index, void* score, void* cluster,
                                                 void* selected_edge, void* cluster_score, void* counts, int64_t nodes,
                                                 int64_t edges, void* workspace, uint64_t workspace_size, void* stream);
}
