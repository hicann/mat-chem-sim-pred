#include "acl/acl.h"
#include "fno_host.h"
#include <cstring>
#include <algorithm>

using namespace optiling;

constexpr int32_t FNO_MAX_CORES = 32;

inline uint32_t AlignUp32(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

extern "C" void aclrtlaunch_fno_forward_kernel(uint32_t blockDim, aclrtStream stream,
                                                void* input, void* weights,
                                                void* output, void* tiling);

static int32_t ComputeCoreCount(int32_t batchSize) {
    if (batchSize <= 0) return 1;
    int32_t cores = std::min((batchSize + 3) / 4, FNO_MAX_CORES);
    return std::max(cores, 1);
}

static int32_t ComputeTotalWeights(int32_t inCh, int32_t hiddenCh,
                                    int32_t outCh, int32_t modes) {
    int32_t lift = inCh * hiddenCh + hiddenCh;
    int32_t spectral = modes * hiddenCh * hiddenCh * 2;
    int32_t project = hiddenCh * outCh + outCh;
    return lift + spectral + project;
}

extern "C" int32_t aclnnFNOForward(
    void* inputAddr, void* weightsAddr, void* outputAddr,
    int32_t batchSize, int32_t gridDim,
    int32_t inChannels, int32_t outChannels,
    int32_t hiddenChannels, int32_t modes,
    void* workspaceAddr, uint64_t workspaceSize, void* stream
) {
    FnoTilingData tiling;
    tiling.batchSize = batchSize;
    tiling.gridDim = gridDim;
    tiling.inChannels = inChannels;
    tiling.outChannels = outChannels;
    tiling.hiddenChannels = hiddenChannels;
    tiling.modes = modes;
    tiling.coreNum = ComputeCoreCount(batchSize);
    tiling.tileSize = gridDim;
    tiling.totalWeightSize = ComputeTotalWeights(inChannels, hiddenChannels,
                                                   outChannels, modes);

    aclrtMemcpyAsync(workspaceAddr, sizeof(FnoTilingData),
                     &tiling, sizeof(FnoTilingData),
                     ACL_MEMCPY_HOST_TO_DEVICE, (aclrtStream)stream);
    aclrtSynchronizeStream((aclrtStream)stream);

    aclrtlaunch_fno_forward_kernel(tiling.coreNum, (aclrtStream)stream,
                                    inputAddr, weightsAddr,
                                    outputAddr, workspaceAddr);
    return 0;
}

extern "C" uint64_t aclnnFNOForwardGetWorkspaceSize(
    int32_t batchSize, int32_t gridDim,
    int32_t inChannels, int32_t outChannels,
    int32_t hiddenChannels, int32_t modes
) {
    return AlignUp32(sizeof(FnoTilingData), 32) + 4096;
}
