#include "mesh_graph_net_kernel.h"

extern "C" __global__ __aicore__ void mesh_graph_net_kernel(
    GM_ADDR nodeFeatures,
    GM_ADDR edgeIndices,
    GM_ADDR edgeFeatures,
    GM_ADDR nodeWeights,
    GM_ADDR edgeWeights,
    GM_ADDR output,
    GM_ADDR tiling
) {
    MeshGraphNetOp op;
    op.Init(nodeFeatures, edgeIndices, edgeFeatures,
             nodeWeights, edgeWeights, output, tiling);
    op.Process();
}
