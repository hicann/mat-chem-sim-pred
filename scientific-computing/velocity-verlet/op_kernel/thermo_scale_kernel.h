/**
 * ThermoScale Kernel: Apply V-rescale + C-rescale scaling to coords/velocities
 *
 * This kernel is launched AFTER the host has computed:
 *   vrescale_lambda — V-rescale stochastic scaling factor
 *   crescale_mu     — C-rescale box scaling factor
 *   v_scale         — mu^(-1/3) for velocity adjustment
 *
 * Algorithm:
 *   If vrescale enabled:  v_i *= lambda   (all atoms)
 *   If barostat enabled:  r_i *= mu, v_i *= v_scale   (all atoms)
 */

#ifndef THERMO_SCALE_KERNEL_H
#define THERMO_SCALE_KERNEL_H

#ifdef VV_DEVICE
#include "kernel_operator.h"
#include "vv_types.h"
using namespace AscendC;

class ThermoScaleKernel {
public:
    __aicore__ ThermoScaleKernel() {}
    __aicore__ ~ThermoScaleKernel() {}

    __aicore__ void Process(VVConfig config, VVKernelGM gm,
                            int32_t tile_start, int32_t tile_end)
    {
        __gm__ float* coords;
        __gm__ float* velocities;

        __builtin_memcpy(&coords,     &gm.coords_gm,     sizeof(coords));
        __builtin_memcpy(&velocities, &gm.velocities_gm, sizeof(velocities));

        float lambda = config.vrescale_lambda;
        float mu = config.crescale_mu;
        float v_scale = config.v_scale;

        bool do_vrescale = (config.enable_vrescale != 0) && (config.tau_t > 0.0f);
        bool do_barostat = (config.enable_barostat != 0) && (config.tau_p > 0.0f);

        for (int32_t i = tile_start; i < tile_end; i++) {
            int32_t i3 = i * 3;

            // V-rescale: v_i *= lambda
            if (do_vrescale) {
                velocities[i3 + 0] *= lambda;
                velocities[i3 + 1] *= lambda;
                velocities[i3 + 2] *= lambda;
            }

            // C-rescale: r_i *= mu, v_i *= v_scale
            if (do_barostat) {
                coords[i3 + 0] *= mu;
                coords[i3 + 1] *= mu;
                coords[i3 + 2] *= mu;
                velocities[i3 + 0] *= v_scale;
                velocities[i3 + 1] *= v_scale;
                velocities[i3 + 2] *= v_scale;
            }
        }
    }
};

#endif // VV_DEVICE
#endif // THERMO_SCALE_KERNEL_H
