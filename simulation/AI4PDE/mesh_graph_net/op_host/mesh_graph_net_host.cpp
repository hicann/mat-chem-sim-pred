#include "acl/acl.h"
#include "mesh_graph_net_host.h"
#include <cstring>
#include <algorithm>

using namespace optiling;

constexpr int32_t MGN_MAX_CORES = 32;

inline uint32_t AlignUp32(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

extern "C" void aclrtlaunch_mesh_graph_net_kernel(
    uint32_t blockDim, aclrtStream stream,
    void* nodeFeatures, void* edgeIndices, void* edgeFeatures,
    void* nodeWeights, void* edgeWeights, void* output,
    void* tiling);

extern "C" int32_t aclnnMeshGraphNet(
    void* nodeFeaturesAddr, void* edgeIndicesAddr,
    void* edgeFeaturesAddr, void* nodeWeightsAddr,
    void* edgeWeightsAddr, void* outputAddr,
    int32_t numNodes, int32_t numEdges,
    int32_t nodeDim, int32_t edgeDim,
    int32_t hiddenDim, int32_t outputDim,
    int32_t maxNeighbors,
    void* workspaceAddr, uint64_t workspaceSize, void* stream
) {
    MeshGraphNetTilingData tiling;
    tiling.numNodes = numNodes;
    tiling.numEdges = numEdges;
    tiling.nodeDim = nodeDim;
    tiling.edgeDim = edgeDim;
    tiling.hiddenDim = hiddenDim;
    tiling.outputDim = outputDim;
    tiling.maxNeighbors = maxNeighbors;
    tiling.tileSize = std::max(nodeDim, edgeDim);
    tiling.coreNum = std::max(1, std::min((numNodes + 31) / 32, MGN_MAX_CORES));
    tiling.numEdgeTiles = (numEdges + 1023) / 1024;
    tiling.nodeWeightSize = (nodeDim * hiddenDim + hiddenDim) + (hiddenDim * hiddenDim + hiddenDim) +
                             (hiddenDim * outputDim + outputDim);
    tiling.edgeWeightSize = ((nodeDim + edgeDim) * hiddenDim + hiddenDim) + (hiddenDim * hiddenDim + hiddenDim);

    aclrtMemcpyAsync(workspaceAddr, sizeof(MeshGraphNetTilingData),
                     &tiling, sizeof(MeshGraphNetTilingData),
                     ACL_MEMCPY_HOST_TO_DEVICE, (aclrtStream)stream);
    aclrtSynchronizeStream((aclrtStream)stream);

    aclrtlaunch_mesh_graph_net_kernel(
        tiling.coreNum, (aclrtStream)stream,
        nodeFeaturesAddr, edgeIndicesAddr, edgeFeaturesAddr,
        nodeWeightsAddr, edgeWeightsAddr, outputAddr,
        workspaceAddr);

    return 0;
}

extern "C" uint64_t aclnnMeshGraphNetGetWorkspaceSize(
    int32_t numNodes, int32_t numEdges,
    int32_t nodeDim, int32_t edgeDim,
    int32_t hiddenDim, int32_t outputDim
) {
    return AlignUp32(sizeof(MeshGraphNetTilingData), 32) + 16384;
}
