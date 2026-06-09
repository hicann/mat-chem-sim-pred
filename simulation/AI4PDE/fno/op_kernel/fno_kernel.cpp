#include "fno_kernel.h"

extern "C" __global__ __aicore__ void fno_forward_kernel(
    GM_ADDR input,
    GM_ADDR weights,
    GM_ADDR output,
    GM_ADDR tiling
) {
    FnoOp op;
    op.Init(input, weights, output, tiling);
    op.Process();
}
