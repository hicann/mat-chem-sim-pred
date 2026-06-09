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
