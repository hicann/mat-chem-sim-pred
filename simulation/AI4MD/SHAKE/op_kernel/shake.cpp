/**
 * shake.cpp
 *
 * SHAKE iteration kernel — Ascend C entry point
 *
 * IMPORTANT: SHAKE_DEVICE must be defined before including shake_types.h
 * to ensure uint64_t layout is used on device side.
 */
#define SHAKE_DEVICE 1
#include "shake_types.h"

#include "kernel_operator.h"
#include "shake_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void shake_iteration(ShakeIterationArgs args) {
    int32_t bidx = block_idx;
    SHAKEIterationKernel kernel;
    kernel.Process(args, bidx);
}
