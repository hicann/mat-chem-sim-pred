#include "pinn_kernel.h"

extern "C" __global__ __aicore__ void pinn_fc_kernel(
    GM_ADDR input,
    GM_ADDR weights,
    GM_ADDR output,
    GM_ADDR gradient,
    GM_ADDR tiling
) {
    PinnFCOp op;
    op.Init(input, weights, output, gradient, tiling);
    op.Process();
}
