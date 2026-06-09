#ifndef PINN_HOST_H
#define PINN_HOST_H

#include <cstdint>

namespace optiling {

struct PinnTilingData {
    int32_t batchSize;
    int32_t inputDim;
    int32_t hiddenDim;
    int32_t outputDim;
    int32_t numLayers;
    int32_t tileSize;
    int32_t coreNum;
    int32_t weightsPerLayer;
    int32_t totalWeightSize;
    int32_t activationType;
};

}  // namespace optiling

extern "C" int32_t aclnnPinnFC(
    void* inputAddr,
    void* weightsAddr,
    void* outputAddr,
    void* gradientAddr,
    int32_t batchSize,
    int32_t inputDim,
    int32_t hiddenDim,
    int32_t outputDim,
    int32_t numLayers,
    int32_t activationType,
    void* workspaceAddr,
    uint64_t workspaceSize,
    void* stream
);

extern "C" uint64_t aclnnPinnFCGetWorkspaceSize(
    int32_t batchSize,
    int32_t inputDim,
    int32_t hiddenDim,
    int32_t outputDim,
    int32_t numLayers
);

#endif
