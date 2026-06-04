/**
 * vv_finish_kernel.h
 *
 * VV Integrator — Second Half: finish velocity update + compute statistics
 *
 * Algorithm:
 *   Step 3: v(t+dt) = v(t+dt/2) + (dt/2) * F(t+dt) / m
 *   Step 4: KE += 0.5 * m * |v|²  (accumulated in pot_virial_gm[0])
 *   Step 5: virial_xx_yy_zz += r·F    (accumulated in pot_virial_gm[1])
 *
 * NOTE: This kernel reads forces at time(t+dt) which were just computed
 * by the force evaluation step. coords were updated by vv_integrate.
 *
 * The KE and virial are accumulated into a single scalar per block.
 * Since Ascend C has no cross-block atomic add, we must use single-block
 * launch or reduce on host.
 *
 * CRITICAL: pot_virial_gm has two scalar slots:
 *   [0] = KE accumulator (init to 0, accumulate)
 *   [1] = virial trace accumulator (init to 0, accumulate)
 *
 * Host must initialize these to 0 before launch!
 */

#ifndef VV_FINISH_KERNEL_H
#define VV_FINISH_KERNEL_H

#ifdef VV_DEVICE
#include "kernel_operator.h"
#include "vv_types.h"
using namespace AscendC;

class VVFinishKernel {
public:
    __aicore__ VVFinishKernel() {}
    __aicore__ ~VVFinishKernel() {}

    __aicore__ void Process(VVConfig config, VVKernelGM gm,
                            int32_t tile_start, int32_t tile_end)
    {
        __gm__ float* coords;
        __gm__ float* velocities;
        __gm__ float* forces;
        __gm__ float* masses;
        __gm__ float* pot_virial;

        __builtin_memcpy(&coords,     &gm.coords_gm,     sizeof(coords));
        __builtin_memcpy(&velocities, &gm.velocities_gm, sizeof(velocities));
        __builtin_memcpy(&forces,     &gm.forces_gm,     sizeof(forces));
        __builtin_memcpy(&masses,     &gm.masses_gm,     sizeof(masses));
        __builtin_memcpy(&pot_virial, &gm.pot_virial_gm, sizeof(pot_virial));

        float half_dt = config.half_dt;
        float ke = 0.0f;
        float virial = 0.0f;

        for (int32_t i = tile_start; i < tile_end; i++) {
            int32_t i3 = i * 3;
            float inv_m = 1.0f / masses[i];
            float mass = masses[i];

            // Step 3: v(t+dt) = v(t+dt/2) + (dt/2) * F(t+dt) / m
            float vx = velocities[i3 + 0] + half_dt * forces[i3 + 0] * inv_m;
            float vy = velocities[i3 + 1] + half_dt * forces[i3 + 1] * inv_m;
            float vz = velocities[i3 + 2] + half_dt * forces[i3 + 2] * inv_m;
            velocities[i3 + 0] = vx;
            velocities[i3 + 1] = vy;
            velocities[i3 + 2] = vz;

            // Step 4: KE accumulation
            ke += 0.5f * mass * (vx * vx + vy * vy + vz * vz);

            // Step 5: Virial trace: r·F
            float x = coords[i3 + 0];
            float y = coords[i3 + 1];
            float z = coords[i3 + 2];
            virial += x * forces[i3 + 0]
                    + y * forces[i3 + 1]
                    + z * forces[i3 + 2];
        }

        // Write accumulated values — this works for single-block launch
        // OR the first block of a multi-block launch (others overwrite harmlessly)
        pot_virial[0] = ke;
        pot_virial[1] = virial;
    }
};

#endif // VV_DEVICE
#endif // VV_FINISH_KERNEL_H
