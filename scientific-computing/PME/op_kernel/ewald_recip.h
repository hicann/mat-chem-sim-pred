/**
 * ewald_recip.h
 *
 * Ewald Reciprocal Space Sum — Ascend C Kernel (AIV)
 *
 * Computes the Ewald reciprocal-space electrostatic contribution
 * by direct summation over reciprocal lattice vectors.
 *
 * This kernel reads existing forces from GM memory, ADDS the
 * Ewald reciprocal + self-energy contributions, and stores results
 * back. It is designed to be called AFTER the main GAFF2 force
 * kernel, which already computed LJ + erfc(αr)/r short-range Coulomb.
 *
 * Ewald reciprocal formula:
 *   E_recip = (2π/V) * Σ_{k≠0} exp(-k²/4α²) * |S(k)|² / k²
 *   S(k) = Σ_i q_i * exp(i·k·r_i)   [structure factor]
 *   F_recip_i = q_i * (4π/V) * Σ k * exp(-k²/4α²) * [...]
 *
 * Self-energy:
 *   E_self = -α/√π * C_coulomb * Σ_i q_i²
 *
 * Note: All operations use union-based bit manipulation for math
 * (same pattern as gaff2_force.h — no <math.h>, no __builtin_*).
 */

#ifndef EWALD_RECIP_H
#define EWALD_RECIP_H

#ifdef EWALD_DEVICE
#include "kernel_operator.h"
#include "ewald_types.h"
using namespace AscendC;
#endif

// ============================================================
// Math Utilities (Ascend C compatible)
// == same as gaff2_force.h patterns ================

// --- Reciprocal (1/x) via Newton-Raphson ---
__aicore__ inline float ewald_recipf(float x) {
    if (x == 0.0f) return 0.0f;
    int32_t i = *(int32_t*)&x;
    i = 0x7EEEEEEE - i;
    float y = *(float*)&i;
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}

// --- Reciprocal sqrt (1/sqrt(x)) ---
__aicore__ inline float ewald_rsqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    int32_t i = *(int32_t*)&x;
    i = 0x5F3759DF - (i >> 1);
    float y = *(float*)&i;
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

// --- Square root ---
__aicore__ inline float ewald_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    float y = ewald_rsqrtf(x);
    return x * y;
}

// --- Cosine via Taylor (range reduced) ---
__aicore__ inline float ewald_cosf(float x) {
    // Reduce to [-π, π]
    const float two_pi = 6.283185307179586f;
    const float inv_two_pi = 0.15915494309189535f;
    float nf = x * inv_two_pi;
    nf = (nf > 0.0f) ? (nf + 0.5f) : (nf - 0.5f);
    int32_t n = (int32_t)nf;
    x = x - (float)n * two_pi;
    const float pi = 3.141592653589793f;
    const float pi_2 = 1.5707963267948966f;
    if (x > pi)  x -= two_pi;
    if (x < -pi) x += two_pi;
    // cos(x) = sin(π/2 - x) for better accuracy near 0
    x = pi_2 - x;
    // sin via Taylor
    if (x > pi)  x -= two_pi;
    if (x < -pi) x += two_pi;
    const float half_pi = pi * 0.5f;
    if (x > half_pi) { x = pi - x; }
    else if (x < -half_pi) { x = -pi - x; }
    float x2 = x * x;
    float x3 = x2 * x;
    float x5 = x2 * x3;
    float x7 = x2 * x2 * x3;
    return x - 0.16666666666666666f * x3 
             + 0.008333333333333333f * x5 
             - 0.0001984126984126984f * x7;
}

// --- Sine via cos(x) = sin(π/2 - x) shift ---
__aicore__ inline float ewald_sinf(float x) {
    const float pi_2 = 1.5707963267948966f;
    return ewald_cosf(pi_2 - x);
}

// --- Exponential: exp(-x) for x >= 0 ---
// exp(-x) = 2^(-x/ln2) = 2^(-n) * 2^(-r) where n = round(x/ln2), r = x/ln2 - n
// 2^(-r) = exp(-r·ln2) via Taylor
__aicore__ inline float ewald_expf_neg(float x) {
    if (x <= 0.0f) return 1.0f;
    if (x > 88.0f) return 0.0f;
    
    const float inv_ln2 = 1.4426950408889634f;
    float nf = x * inv_ln2;
    int32_t n = (int32_t)(nf + 0.5f);
    float r = nf - (float)n;
    
    // 2^(-r) = exp(-r * ln2) via Taylor
    const float ln2 = 0.6931471805599453f;
    float z = -r * ln2;
    float exp2mr = 1.0f + z * (1.0f + z * (0.5f + z * (0.16666666666666666f + z * 0.041666666666666664f)));
    
    if (n > 127) return 0.0f;
    
    int32_t exp_bits = (127 - n) << 23;
    float exp_factor;
    int32_t* ep = (int32_t*)&exp_factor;
    *ep = exp_bits;
    
    return exp2mr * exp_factor;
}

// ============================================================
// Ewald Kernel Class
// ============================================================

class EwaldRecipKernel {
public:
    __aicore__ EwaldRecipKernel() {}
    __aicore__ ~EwaldRecipKernel() {}

    // ============================================================
    // ENTRY POINT
    // ============================================================
    __aicore__ void Process(
        EwaldConfig config,
        EwaldKernelGM gm,
        int32_t tile_start,
        int32_t tile_end)
    {
        // Decode GM pointers
        __gm__ float* coords;
        __gm__ int32_t* types;
        __gm__ float* type_params;
        __gm__ float* forces_gm;
        __gm__ float* pot;
        
        __builtin_memcpy(&coords, &gm.coords_gm, sizeof(coords));
        __builtin_memcpy(&types, &gm.types_gm, sizeof(types));
        __builtin_memcpy(&type_params, &gm.type_params_gm, sizeof(type_params));
        __builtin_memcpy(&forces_gm, &gm.forces_gm, sizeof(forces_gm));
        __builtin_memcpy(&pot, &gm.pot_gm, sizeof(pot));
        
        int32_t n_atoms = config.n_atoms;
        float box_size  = config.box_size;
        float alpha     = config.ewald_alpha;
        float volume    = config.volume;
        int32_t n_max   = (int32_t)(config.kmax + 0.5f);
        
        const float COULOMB_C = 138.935458f;
        const float inv_sqrt_pi = 0.5641895835477563f;
        const float two_pi = 6.283185307179586f;
        
        float k_min = two_pi / box_size;
        float prefac = two_pi / volume;  // = 2π/V
        
        // ============================================================
        // Reciprocal sum over k-vectors
        // ============================================================
        float e_recip = 0.0f;
        
        for (int32_t nx = -n_max; nx <= n_max; nx++) {
            for (int32_t ny = -n_max; ny <= n_max; ny++) {
                for (int32_t nz = -n_max; nz <= n_max; nz++) {
                    if (nx == 0 && ny == 0 && nz == 0) continue;
                    
                    float kx = k_min * (float)nx;
                    float ky = k_min * (float)ny;
                    float kz = k_min * (float)nz;
                    float k2 = kx * kx + ky * ky + kz * kz;
                    if (k2 < 1e-20f) continue;
                    
                    // Structure factor S(k) = Σ qi * exp(i k·r)
                    float S_re = 0.0f, S_im = 0.0f;
                    for (int32_t i = 0; i < n_atoms; i++) {
                        float xi = coords[i * 3 + 0];
                        float yi = coords[i * 3 + 1];
                        float zi = coords[i * 3 + 2];
                        int32_t ti = types[i];
                        float qi = type_params[ti * 3 + 2];
                        float kr = kx * xi + ky * yi + kz * zi;
                        S_re += qi * ewald_cosf(kr);
                        S_im += qi * ewald_sinf(kr);
                    }
                    
                    // Gaussian factor: exp(-k²/4α²)
                    float inv_4alpha2 = 0.25f / (alpha * alpha);
                    float gaussian = ewald_expf_neg(k2 * inv_4alpha2);
                    float inv_k2 = ewald_recipf(k2);
                    float S_sq = S_re * S_re + S_im * S_im;
                    
                    e_recip += prefac * inv_k2 * gaussian * S_sq;
                    
                    // ============================================================
                    // Forces (ADDITIVE to existing forces from GAFF2 kernel)
                    // F_i = q_i * (4π*C/V) * Σ (k/k²) * exp(-k²/4α²) * [S_re·sin - S_im·cos]
                    // Note: C_coulomb (COULOMB_C) is NOT in the energy prefac (it's
                    // already in the formula because S(k) contains qi in e, and the
                    // Coulomb constant must multiply the final result).
                    // 
                    // Actually: the energy E = COULOMB_C * (2π/V) * Σ exp(...)
                    // Force: deriv of E w.r.t r_i brings factor of 2 from |S|²
                    // F_i = COULOMB_C * (4π/V) * Σ q_i * k * exp(...) * [S_re·sin - S_im·cos] / k²
                    // ============================================================
                    float f_pre = COULOMB_C * two_pi * 2.0f * inv_k2 * gaussian / volume;
                    
                    for (int32_t i = 0; i < n_atoms; i++) {
                        float xi = coords[i * 3 + 0];
                        float yi = coords[i * 3 + 1];
                        float zi = coords[i * 3 + 2];
                        int32_t ti = types[i];
                        float qi = type_params[ti * 3 + 2];
                        float kr = kx * xi + ky * yi + kz * zi;
                        float sk = ewald_sinf(kr);
                        float ck = ewald_cosf(kr);
                        
                        float amp = qi * f_pre * (S_re * sk - S_im * ck);
                        
                        forces_gm[i * 3 + 0] += amp * kx;
                        forces_gm[i * 3 + 1] += amp * ky;
                        forces_gm[i * 3 + 2] += amp * kz;
                    }
                }
            }
        }
        
        // ============================================================
        // Self-energy correction
        // E_self = -α/√π * COULOMB_C * Σ q_i²
        // ============================================================
        float self_sum = 0.0f;
        for (int32_t i = 0; i < n_atoms; i++) {
            int32_t ti = types[i];
            float qi = type_params[ti * 3 + 2];
            self_sum += qi * qi;
        }
        float e_self = -alpha * inv_sqrt_pi * COULOMB_C * self_sum;
        
        // Store results
        pot[0] = e_recip;
        pot[1] = e_self;
        pot[2] = e_recip + e_self;
    }
};

#endif // EWALD_RECIP_H
