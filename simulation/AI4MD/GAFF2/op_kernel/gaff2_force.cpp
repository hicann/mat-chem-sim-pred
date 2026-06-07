/**
 * gaff2_force.cpp
 *
 * GAFF2 Force Field — Ascend C Kernel Entry Point
 *
 * Entry point function: gaff2_compute_forces
 *   - Called by host via aclrtlaunch_gaff2_force
 *   - Single AIV kernel processes all bonds, angles, dihedrals, nonbonded
 *
 * IMPORTANT: Entry points must be declared ONLY in .cpp files, NOT in .h files.
 * The auto-gen tool (gen_maia_kernel.py) parses .cpp and generates wrapper code.
 * Declaring entries in .h causes "redefinition" errors.
 */

// Include types BEFORE FORCE_TYPES_DEVICE so auto-gen host_stub sees them
#include "gaff2_types.h"

#define FORCE_TYPES_DEVICE 1

#include "kernel_operator.h"
// gaff2_types.h was already included above; the GAFF2_DEVICE guard ensures
// device-specific constructs are only used when FORCE_TYPES_DEVICE is defined.
// We undef/redef to allow the host_stub to have type definitions without __gm__.
#undef GAFF2_DEVICE
#define GAFF2_DEVICE 1
#include "gaff2_types.h"  // Now with __gm__ support
#include "gaff2_force.h"

using namespace AscendC;

// ============================================================
// Kernel Entry Point
// ============================================================
// Revert to original: no __builtin_memcpy
extern "C" __global__ __aicore__ void gaff2_compute_forces(
    GAFF2ForceArgs args)
{
    // Get the block index
    int32_t bidx = block_idx;
    int32_t bnum = block_num;
    
    // GAFF2ForceKernel instance (per-block, zero-initialized)
    GAFF2ForceKernel kernel;
    
    // Determine which atoms this block handles
    int32_t n_atoms = args.config.n_atoms;
    int32_t atoms_per_block = (n_atoms + bnum - 1) / bnum;
    int32_t tile_start = bidx * atoms_per_block;
    int32_t tile_end = (bidx + 1) * atoms_per_block;
    if (tile_end > n_atoms) tile_end = n_atoms;
    
    // Process all interactions for this tile
    kernel.Process(args.config, args.gm, tile_start, tile_end);
}
