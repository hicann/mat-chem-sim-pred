/**
 * shake_kernel.h
 *
 * SHAKE iteration kernel — Ascend C Implementation (Serial)
 *
 * Performs ONE iteration of SHAKE constraint solving on NPU.
 * All constraints are processed serially within a single block.
 */
#ifndef SHAKE_KERNEL_H
#define SHAKE_KERNEL_H

#ifdef SHAKE_DEVICE

#include "kernel_operator.h"
#include "shake_types.h"
using namespace AscendC;

class SHAKEIterationKernel {
public:
    __aicore__ SHAKEIterationKernel() {}
    __aicore__ ~SHAKEIterationKernel() {}

    __aicore__ void Process(ShakeIterationArgs& args, int32_t bidx) {
        if (bidx != 0) return;

        __gm__ float* coords;
        __gm__ int32_t* constraints;
        __gm__ float* target_lengths;
        __gm__ float* inv_masses;
        __gm__ float* max_error;

        __builtin_memcpy(&coords, &args.coords_gm, sizeof(coords));
        __builtin_memcpy(&constraints, &args.constraints_gm, sizeof(constraints));
        __builtin_memcpy(&target_lengths, &args.target_lengths_gm, sizeof(target_lengths));
        __builtin_memcpy(&inv_masses, &args.inv_masses_gm, sizeof(inv_masses));
        __builtin_memcpy(&max_error, &args.max_error_gm, sizeof(max_error));

        int32_t n_constraints = args.n_constraints;
        float dt_sq = args.dt_sq;
        float tolerance = args.tolerance;

        float max_err_val = 0.0f;

        for (int32_t c = 0; c < n_constraints; c++) {
            int32_t i = constraints[c * 2 + 0];
            int32_t j = constraints[c * 2 + 1];

            float xi = coords[i * 3 + 0];
            float yi = coords[i * 3 + 1];
            float zi = coords[i * 3 + 2];
            float xj = coords[j * 3 + 0];
            float yj = coords[j * 3 + 1];
            float zj = coords[j * 3 + 2];

            float dx = xi - xj;
            float dy = yi - yj;
            float dz = zi - zj;
            float r_sq = dx * dx + dy * dy + dz * dz;

            float d0 = target_lengths[c];
            float d0_sq = d0 * d0;

            float error = r_sq - d0_sq;
            float abs_err = (error >= 0.0f) ? error : -error;

            if (abs_err > max_err_val) max_err_val = abs_err;

            if (abs_err < tolerance) continue;

            float inv_mi = inv_masses[i];
            float inv_mj = inv_masses[j];
            float inv_m_sum = inv_mi + inv_mj;

            float denominator = 2.0f * dt_sq * inv_m_sum * r_sq;
            if (denominator < 1e-30f) continue;

            float lambda = error / denominator;

            float corr_scale_i = lambda * dt_sq * inv_mi;
            float corr_scale_j = lambda * dt_sq * inv_mj;

            coords[i * 3 + 0] = xi - corr_scale_i * dx;
            coords[i * 3 + 1] = yi - corr_scale_i * dy;
            coords[i * 3 + 2] = zi - corr_scale_i * dz;
            coords[j * 3 + 0] = xj + corr_scale_j * dx;
            coords[j * 3 + 1] = yj + corr_scale_j * dy;
            coords[j * 3 + 2] = zj + corr_scale_j * dz;
        }

        max_error[0] = max_err_val;
    }
};

#endif // SHAKE_DEVICE
#endif // SHAKE_KERNEL_H
