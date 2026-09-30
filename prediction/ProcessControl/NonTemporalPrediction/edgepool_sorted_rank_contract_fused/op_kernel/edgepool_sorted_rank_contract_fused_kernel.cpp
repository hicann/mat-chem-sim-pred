/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include "edgepool_sorted_rank_contract_fused_impl.h"

extern "C" __global__ __aicore__ void edgepool_sorted_rank_contract_fused_kernel(
    GM_ADDR order, GM_ADDR edgeIndex, GM_ADDR score, GM_ADDR cluster, GM_ADDR selectedEdge, GM_ADDR clusterScore,
    GM_ADDR counts, GM_ADDR workspace, EdgepoolSortedRankContractTiling tiling)
{
    (void)workspace;
    EdgepoolSortedRankContractKernel kernel;
    kernel.Init(order, edgeIndex, score, cluster, selectedEdge, clusterScore, counts, tiling);
    kernel.Process();
}
