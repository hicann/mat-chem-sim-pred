#ifndef DEEPONET_HOST_H
#define DEEPONET_HOST_H

#include <cstdint>

namespace optiling {

struct DeepOnetTilingData {
    int32_t batchSize;
    int32_t branchDim;
    int32_t trunkDim;
    int32_t latentDim;
    int32_t querySize;
    int32_t tileSize;
    int32_t coreNum;
    int32_t branchWeightSize;
    int32_t trunkWeightSize;
};

}  // namespace optiling

extern "C" int32_t aclnnDeepOnetForward(
    void* branchInputAddr,
    void* trunkInputAddr,
    void* branchWeightsAddr,
    void* trunkWeightsAddr,
    void* outputAddr,
    int32_t batchSize,
    int32_t branchDim,
    int32_t trunkDim,
    int32_t latentDim,
    int32_t querySize,
    void* workspaceAddr,
    uint64_t workspaceSize,
    void* stream
);

extern "C" uint64_t aclnnDeepOnetForwardGetWorkspaceSize(
    int32_t batchSize,
    int32_t branchDim,
    int32_t trunkDim,
    int32_t latentDim,
    int32_t querySize
);

#endif
