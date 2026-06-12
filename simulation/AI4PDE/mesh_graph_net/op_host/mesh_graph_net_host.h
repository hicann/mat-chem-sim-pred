/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MESH_GRAPH_NET_HOST_H
#define MESH_GRAPH_NET_HOST_H

#include <cstdint>

namespace optiling {

struct MeshGraphNetTilingData {
    int32_t numNodes;
    int32_t numEdges;
    int32_t nodeDim;
    int32_t edgeDim;
    int32_t hiddenDim;
    int32_t outputDim;
    int32_t maxNeighbors;
    int32_t tileSize;
    int32_t coreNum;
    int32_t numEdgeTiles;
    int32_t nodeWeightSize;
    int32_t edgeWeightSize;
};

}  // namespace optiling

extern "C" int32_t aclnnMeshGraphNet(
    void* nodeFeaturesAddr,
    void* edgeIndicesAddr,
    void* edgeFeaturesAddr,
    void* nodeWeightsAddr,
    void* edgeWeightsAddr,
    void* outputAddr,
    int32_t numNodes,
    int32_t numEdges,
    int32_t nodeDim,
    int32_t edgeDim,
    int32_t hiddenDim,
    int32_t outputDim,
    int32_t maxNeighbors,
    void* workspaceAddr,
    uint64_t workspaceSize,
    void* stream
);

extern "C" uint64_t aclnnMeshGraphNetGetWorkspaceSize(
    int32_t numNodes,
    int32_t numEdges,
    int32_t nodeDim,
    int32_t edgeDim,
    int32_t hiddenDim,
    int32_t outputDim
);

#endif
