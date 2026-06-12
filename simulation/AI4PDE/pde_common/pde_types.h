/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PDE_TYPES_H
#define PDE_TYPES_H

#include <cstdint>

namespace pde {

struct FCTilingData {
    int32_t inputDim;
    int32_t hiddenDim;
    int32_t outputDim;
    int32_t numLayers;
    int32_t batchSize;
    int32_t tileSize;
    int32_t coreNum;
    float activationParam;
};

struct FNOTilingData {
    int32_t gridDimX;
    int32_t gridDimY;
    int32_t modesX;
    int32_t modesY;
    int32_t inChannels;
    int32_t outChannels;
    int32_t hiddenChannels;
    int32_t tileSize;
    int32_t coreNum;
};

struct DeepONetTilingData {
    int32_t branchDim;
    int32_t trunkDim;
    int32_t latentDim;
    int32_t batchSize;
    int32_t querySize;
    int32_t tileSize;
    int32_t coreNum;
};

struct MeshGraphNetTilingData {
    int32_t numNodes;
    int32_t numEdges;
    int32_t nodeDim;
    int32_t edgeDim;
    int32_t hiddenDim;
    int32_t maxNeighbors;
    int32_t tileSize;
    int32_t coreNum;
    int32_t numEdgeTiles;
};

}  // namespace pde

#endif
