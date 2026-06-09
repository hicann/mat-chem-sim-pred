#include "acl/acl.h"
#include "pinn_host.h"
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace optiling;

constexpr int32_t PINN_MAX_CORES = 32;

inline uint32_t AlignUp32(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

extern "C" void aclrtlaunch_pinn_fc_kernel(uint32_t blockDim, aclrtStream stream,
                                            void* input, void* weights,
                                            void* output, void* gradient,
                                            void* tiling);

static int32_t ComputeCoreCount(int32_t batchSize) {
    if (batchSize <= 0) return 1;
    int32_t cores = (batchSize + 31) / 32;
    cores = std::min(cores, PINN_MAX_CORES);
    cores = std::max(cores, 1);
    return cores;
}

static int32_t ComputeTileSize(int32_t batchSize, int32_t coreNum) {
    int32_t ptsPerCore = (batchSize + coreNum - 1) / coreNum;
    int32_t tile = ptsPerCore;
    if (tile > 64) tile = 64;
    if (tile < 1) tile = 1;
    return tile;
}

static int32_t ComputeTotalWeights(int32_t inputDim, int32_t hiddenDim,
                                    int32_t outputDim, int32_t numLayers) {
    int32_t total = 0;
    int32_t inDim = inputDim;
    for (int32_t l = 0; l < numLayers; l++) {
        int32_t outDim = (l == numLayers - 1) ? outputDim : hiddenDim;
        total += inDim * outDim + outDim;
        inDim = outDim;
    }
    return total;
}

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
) {
    PinnTilingData tiling;
    tiling.batchSize = batchSize;
    tiling.inputDim = inputDim;
    tiling.hiddenDim = hiddenDim;
    tiling.outputDim = outputDim;
    tiling.numLayers = numLayers;
    tiling.activationType = activationType;
    tiling.totalWeightSize = ComputeTotalWeights(inputDim, hiddenDim, outputDim, numLayers);
    tiling.coreNum = ComputeCoreCount(batchSize);
    tiling.tileSize = ComputeTileSize(batchSize, tiling.coreNum);

    aclrtMemcpyAsync(workspaceAddr, sizeof(PinnTilingData),
                     &tiling, sizeof(PinnTilingData),
                     ACL_MEMCPY_HOST_TO_DEVICE, (aclrtStream)stream);
    aclrtSynchronizeStream((aclrtStream)stream);

    aclrtlaunch_pinn_fc_kernel(tiling.coreNum, (aclrtStream)stream,
                                inputAddr, weightsAddr,
                                outputAddr, gradientAddr,
                                workspaceAddr);

    return 0;
}

extern "C" uint64_t aclnnPinnFCGetWorkspaceSize(
    int32_t batchSize,
    int32_t inputDim,
    int32_t hiddenDim,
    int32_t outputDim,
    int32_t numLayers
) {
    return AlignUp32(sizeof(PinnTilingData), 32) + 1024;
}
