/**
 * ewald_kernel.cpp
 *
 * Ascend C kernel entry point for Ewald reciprocal sum.
 *
 * Entry point: ewald_reciprocal(EwaldForceArgs args)
 * The args struct is passed by value (auto-gen host_stub handles encoding).
 */

#define EWALD_DEVICE 1
#include "kernel_operator.h"
#include "ewald_types.h"
#include "ewald_recip.h"

using namespace AscendC;

// ============================================================
// Kernel entry point — takes EwaldForceArgs by value
// ============================================================
extern "C" __global__ __aicore__ void ewald_reciprocal(EwaldForceArgs args) {
    EwaldRecipKernel kernel;
    kernel.Process(args.config, args.gm, args.tile_start, args.tile_end);
}
