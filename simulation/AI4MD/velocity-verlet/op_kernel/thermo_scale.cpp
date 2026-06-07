/**
 * thermo_scale.cpp
 *
 * ThermoScale — Ascend C Kernel Entry Point
 *
 * Performs V-rescale velocity scaling and/or C-rescale coordinate/velocity scaling.
 */

#define VV_DEVICE 1
#include "kernel_operator.h"
#include "vv_types.h"
#include "thermo_scale_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void thermo_scale_kernel(VVForceArgs args) {
    int32_t bidx = block_idx;
    int32_t bnum = block_num;
    int32_t n_atoms = args.config.n_atoms;
    int32_t atoms_per_block = (n_atoms + bnum - 1) / bnum;
    int32_t tile_start = bidx * atoms_per_block;
    int32_t tile_end = (bidx + 1) * atoms_per_block;
    if (tile_end > n_atoms) tile_end = n_atoms;

    ThermoScaleKernel kernel;
    kernel.Process(args.config, args.gm, tile_start, tile_end);
}
