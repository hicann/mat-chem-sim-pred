#include "deeponet_kernel.h"

extern "C" __global__ __aicore__ void deeponet_forward_kernel(
    GM_ADDR branchInput,
    GM_ADDR trunkInput,
    GM_ADDR branchWeights,
    GM_ADDR trunkWeights,
    GM_ADDR output,
    GM_ADDR tiling
) {
    DeepOnetOp op;
    op.Init(branchInput, trunkInput, branchWeights, trunkWeights, output, tiling);
    op.Process();
}
