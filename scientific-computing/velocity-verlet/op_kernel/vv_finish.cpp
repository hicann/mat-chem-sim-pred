/**
 * vv_finish.cpp
 *
 * VV Finish — Ascend C Kernel Entry Point
 *
 * Performs: v(t+dt) = v(t+dt/2) + (dt/2)*F(t+dt)/m
 *           KE += 0.5*m*|v|²
 *           virial += r·F
 */

#define VV_DEVICE 1
#include "kernel_operator.h"
#include "vv_types.h"
#include "vv_finish_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void vv_finish_kernel(VVForceArgs args) {
    int32_t bidx = block_idx;
    int32_t bnum = block_num;
    int32_t n_atoms = args.config.n_atoms;
    int32_t atoms_per_block = (n_atoms + bnum - 1) / bnum;
    int32_t tile_start = bidx * atoms_per_block;
    int32_t tile_end = (bidx + 1) * atoms_per_block;
    if (tile_end > n_atoms) tile_end = n_atoms;

    VVFinishKernel kernel;
    kernel.Process(args.config, args.gm, tile_start, tile_end);
}
