/**
 * pme_kernel.cpp
 *
 * PME (Particle Mesh Ewald) 3D FFT — Ascend C Kernel Entry Point
 *
 * Entry point: pme_3d_fft(PMEArgs args)
 *
 * This kernel replaces the direct Ewald reciprocal sum for systems
 * where O(kmax³·N) becomes prohibitive. Instead, it uses:
 *   1. B-spline charge spreading (O(N))
 *   2. 3D FFT via vectorized DFT matrix multiply (O(M³·M))
 *   3. Influence function multiplication (O(M³))
 *   4. 3D IFFT (O(M³·M))
 *   5. Force interpolation via B-spline gradient (O(N))
 *
 * Total complexity: O(N + M⁴) where M=mesh_dim (~32-64)
 * vs direct Ewald O(kmax³·N) where kmax~8.
 */

#define PME_DEVICE 1
#include "kernel_operator.h"
#include "pme_types.h"
#include "pme_fft3d.h"

using namespace AscendC;

// ============================================================
// Kernel entry point — takes PMEArgs by value
// ============================================================
extern "C" __global__ __aicore__ void pme_3d_fft(PMEArgs args) {
    PMEFFT3DKernel kernel;
    kernel.Process(args.config, args.gm, args.tile_start, args.tile_end);
}
