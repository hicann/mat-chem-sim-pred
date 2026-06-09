#ifndef FNO_HOST_H
#define FNO_HOST_H

#include <cstdint>

namespace optiling {

struct FnoTilingData {
    int32_t batchSize;
    int32_t gridDim;
    int32_t inChannels;
    int32_t outChannels;
    int32_t hiddenChannels;
    int32_t modes;
    int32_t tileSize;
    int32_t coreNum;
    int32_t totalWeightSize;
};

}  // namespace optiling

extern "C" int32_t aclnnFNOForward(
    void* inputAddr,
    void* weightsAddr,
    void* outputAddr,
    int32_t batchSize,
    int32_t gridDim,
    int32_t inChannels,
    int32_t outChannels,
    int32_t hiddenChannels,
    int32_t modes,
    void* workspaceAddr,
    uint64_t workspaceSize,
    void* stream
);

extern "C" uint64_t aclnnFNOForwardGetWorkspaceSize(
    int32_t batchSize,
    int32_t gridDim,
    int32_t inChannels,
    int32_t outChannels,
    int32_t hiddenChannels,
    int32_t modes
);

#endif
