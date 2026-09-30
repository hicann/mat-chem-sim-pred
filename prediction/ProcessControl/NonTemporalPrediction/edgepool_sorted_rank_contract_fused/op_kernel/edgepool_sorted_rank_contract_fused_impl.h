/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef EDGEPOOL_SORTED_RANK_CONTRACT_FUSED_IMPL_H
#define EDGEPOOL_SORTED_RANK_CONTRACT_FUSED_IMPL_H

#include "kernel_operator.h"

using namespace AscendC;

struct EdgepoolSortedRankContractTiling
{
    uint32_t nodes;
    uint32_t edges;
};

namespace
{

class EdgepoolSortedRankContractKernel
{
   public:
    __aicore__ inline EdgepoolSortedRankContractKernel() = default;

    __aicore__ inline void Init(GM_ADDR order, GM_ADDR edgeIndex, GM_ADDR score, GM_ADDR cluster, GM_ADDR selectedEdge,
                                GM_ADDR clusterScore, GM_ADDR counts, const EdgepoolSortedRankContractTiling& tiling)
    {
        nodes_ = tiling.nodes;
        edges_ = tiling.edges;
        orderGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(order), edges_);
        edgeIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(edgeIndex), static_cast<uint64_t>(edges_) * 2U);
        scoreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(score), edges_);
        clusterGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cluster), nodes_);
        selectedEdgeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selectedEdge), nodes_);
        clusterScoreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(clusterScore), nodes_);
        countsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts), 2U);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0U)
        {
            return;
        }
        for (uint32_t node = 0U; node < nodes_; ++node)
        {
            clusterGm_.SetValue(node, -1);
        }
        const uint32_t selected = SelectEdges();

        uint32_t clusters = selected;
        for (uint32_t node = 0U; node < nodes_; ++node)
        {
            if (clusterGm_.GetValue(node) == -1)
            {
                clusterGm_.SetValue(node, static_cast<int32_t>(clusters));
                clusterScoreGm_.SetValue(clusters, 1.0F);
                ++clusters;
            }
        }
        countsGm_.SetValue(0U, static_cast<int32_t>(selected));
        countsGm_.SetValue(1U, static_cast<int32_t>(clusters));
    }

   private:
    __aicore__ inline uint32_t SelectEdges()
    {
        uint32_t selected = 0U;
        for (uint32_t rank = 0U; rank < edges_; ++rank)
        {
            const int32_t edge = orderGm_.GetValue(rank);
            const bool validEdge = edge >= 0 && static_cast<uint32_t>(edge) < edges_;
            if (!validEdge)
            {
                continue;
            }
            const uint32_t edgeId = static_cast<uint32_t>(edge);
            const int32_t source = edgeIndexGm_.GetValue(edgeId);
            const int32_t target = edgeIndexGm_.GetValue(edges_ + edgeId);
            const bool validNodes = source >= 0 && target >= 0 && static_cast<uint32_t>(source) < nodes_ &&
                                    static_cast<uint32_t>(target) < nodes_;
            if (!validNodes)
            {
                continue;
            }
            const uint32_t sourceId = static_cast<uint32_t>(source);
            const uint32_t targetId = static_cast<uint32_t>(target);
            if (clusterGm_.GetValue(sourceId) != -1 || clusterGm_.GetValue(targetId) != -1)
            {
                continue;
            }
            const int32_t cluster = static_cast<int32_t>(selected);
            clusterGm_.SetValue(sourceId, cluster);
            if (sourceId != targetId)
            {
                clusterGm_.SetValue(targetId, cluster);
            }
            clusterScoreGm_.SetValue(selected, scoreGm_.GetValue(edgeId));
            selectedEdgeGm_.SetValue(selected, edge);
            ++selected;
        }

        return selected;
    }

    GlobalTensor<int32_t> orderGm_;
    GlobalTensor<int32_t> edgeIndexGm_;
    GlobalTensor<float> scoreGm_;
    GlobalTensor<int32_t> clusterGm_;
    GlobalTensor<int32_t> selectedEdgeGm_;
    GlobalTensor<float> clusterScoreGm_;
    GlobalTensor<int32_t> countsGm_;
    uint32_t nodes_ = 0U;
    uint32_t edges_ = 0U;
};

}  // namespace

#endif  // EDGEPOOL_SORTED_RANK_CONTRACT_FUSED_IMPL_H
