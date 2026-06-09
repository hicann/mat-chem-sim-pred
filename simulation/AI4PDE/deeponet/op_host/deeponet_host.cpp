#include "acl/acl.h"
#include "deeponet_host.h"
#include <cstring>
#include <algorithm>

using namespace optiling;

constexpr int32_t DON_MAX_CORES = 32;

inline uint32_t AlignUp32(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

extern "C" void aclrtlaunch_deeponet_forward_kernel(
    uint32_t blockDim, aclrtStream stream,
    void* branchInput, void* trunkInput,
    void* branchWeights, void* trunkWeights,
    void* output, void* tiling);

extern "C" int32_t aclnnDeepOnetForward(
    void* branchInputAddr, void* trunkInputAddr,
    void* branchWeightsAddr, void* trunkWeightsAddr,
    void* outputAddr,
    int32_t batchSize, int32_t branchDim,
    int32_t trunkDim, int32_t latentDim,
    int32_t querySize,
    void* workspaceAddr, uint64_t workspaceSize, void* stream
) {
    DeepOnetTilingData tiling;
    tiling.batchSize = batchSize;
    tiling.branchDim = branchDim;
    tiling.trunkDim = trunkDim;
    tiling.latentDim = latentDim;
    tiling.querySize = querySize;
    tiling.tileSize = std::max(branchDim, trunkDim);
    tiling.coreNum = std::max(1, std::min((batchSize + 3) / 4, DON_MAX_CORES));
    tiling.branchWeightSize = branchDim * latentDim + latentDim;
    tiling.trunkWeightSize = trunkDim * latentDim + latentDim;

    aclrtMemcpyAsync(workspaceAddr, sizeof(DeepOnetTilingData),
                     &tiling, sizeof(DeepOnetTilingData),
                     ACL_MEMCPY_HOST_TO_DEVICE, (aclrtStream)stream);
    aclrtSynchronizeStream((aclrtStream)stream);

    aclrtlaunch_deeponet_forward_kernel(
        tiling.coreNum, (aclrtStream)stream,
        branchInputAddr, trunkInputAddr,
        branchWeightsAddr, trunkWeightsAddr,
        outputAddr, workspaceAddr);

    return 0;
}

extern "C" uint64_t aclnnDeepOnetForwardGetWorkspaceSize(
    int32_t batchSize, int32_t branchDim,
    int32_t trunkDim, int32_t latentDim,
    int32_t querySize
) {
    return AlignUp32(sizeof(DeepOnetTilingData), 32) + 4096;
}
