/**
 * vv_integrate_kernel.h
 *
 * VV Integrator — First Half: velocity half-step + position update
 *
 * Algorithm (one kernel):
 *   v(t+dt/2) = v(t) + (dt/2) * F(t) / m    [half-step velocity]
 *   r(t+dt)   = r(t) + dt * v(t+dt/2)        [position update with PBC]
 *
 * Both operations are O(n) and element-independent per atom.
 * This kernel is launched AFTER forces are computed at coords(t).
 */

#ifndef VV_INTEGRATE_KERNEL_H
#define VV_INTEGRATE_KERNEL_H

#ifdef VV_DEVICE
#include "kernel_operator.h"
#include "vv_types.h"
using namespace AscendC;

// ============================================================
// VV Integrate First Half Kernel
// ============================================================
class VVIntegrateKernel {
public:
    __aicore__ VVIntegrateKernel() {}
    __aicore__ ~VVIntegrateKernel() {}

    __aicore__ void Process(VVConfig config, VVKernelGM gm,
                            int32_t tile_start, int32_t tile_end)
    {
        // Decode GM pointers
        __gm__ float* coords;
        __gm__ float* velocities;
        __gm__ float* forces;
        __gm__ float* masses;

        __builtin_memcpy(&coords,     &gm.coords_gm,     sizeof(coords));
        __builtin_memcpy(&velocities, &gm.velocities_gm, sizeof(velocities));
        __builtin_memcpy(&forces,     &gm.forces_gm,     sizeof(forces));
        __builtin_memcpy(&masses,     &gm.masses_gm,     sizeof(masses));

        float half_dt = config.half_dt;
        float dt = config.dt;
        float box_size = config.box_size;
        float half_box = config.half_box;

        for (int32_t i = tile_start; i < tile_end; i++) {
            int32_t i3 = i * 3;
            float inv_m = 1.0f / masses[i];

            // Step 1: v(t+dt/2) = v(t) + (dt/2) * F(t) / m
            float vx = velocities[i3 + 0] + half_dt * forces[i3 + 0] * inv_m;
            float vy = velocities[i3 + 1] + half_dt * forces[i3 + 1] * inv_m;
            float vz = velocities[i3 + 2] + half_dt * forces[i3 + 2] * inv_m;
            velocities[i3 + 0] = vx;
            velocities[i3 + 1] = vy;
            velocities[i3 + 2] = vz;

            // Step 2: r(t+dt) = r(t) + dt * v(t+dt/2) with PBC
            float x = coords[i3 + 0] + dt * vx;
            float y = coords[i3 + 1] + dt * vy;
            float z = coords[i3 + 2] + dt * vz;

            // PBC wrap into [0, box_size)
            // Use reciprocal mul instead of div for performance
            const float inv_box = 1.0f / box_size;
            float nx = x * inv_box;
            float ny = y * inv_box;
            float nz = z * inv_box;
            nx = (nx >= 0.0f) ? (nx + 1e-10f) : (nx - 1e-10f);
            ny = (ny >= 0.0f) ? (ny + 1e-10f) : (ny - 1e-10f);
            nz = (nz >= 0.0f) ? (nz + 1e-10f) : (nz - 1e-10f);
            int32_t ix = (int32_t)nx;
            int32_t iy = (int32_t)ny;
            int32_t iz = (int32_t)nz;
            coords[i3 + 0] = x - (float)ix * box_size;
            coords[i3 + 1] = y - (float)iy * box_size;
            coords[i3 + 2] = z - (float)iz * box_size;
        }
    }
};

#endif // VV_DEVICE
#endif // VV_INTEGRATE_KERNEL_H
