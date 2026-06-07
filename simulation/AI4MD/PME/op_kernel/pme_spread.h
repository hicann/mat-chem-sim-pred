/**
 * pme_spread.h
 *
 * PME Charge Spreading and Force Interpolation — Ascend C Kernel
 *
 * Two major functions:
 *   1. SpreadCharges: q_i → Q(m) via B-spline interpolation
 *      Each atom's charge is distributed to M³ grid points
 *      using n-th order B-splines.
 *
 *   2. InterpolateForces: F(m) → F_i via B-spline gradient
 *      The reciprocal-space forces on the grid are interpolated
 *      back to atomic positions using B-spline derivatives.
 *
 * B-spline formulas (cubic, n=4):
 *   θ₀(u) = 1          (0 ≤ u < 1)
 *   θ₁(u) = u          (0 ≤ u < 1)  (piecewise constant)
 *   θ₂(u) = ½·(u+1)² - 3/2·(u-½)²  (linear)
 *   θ₃(u) = 1/6·u³     (quadratic)
 *
 * For cubic B-spline (n=4), we use closed-form:
 *   b = 1/6 - ½·u²·(1 - u/3)                     |u| < 1
 *   b = 1/3·(1 - |u|)³
 *
 * Gradient (for force interpolation):
 *   db/du = -u·(1 - u/2)                         |u| < 1
 *   db/du = -(1 - |u|)²·sign(u)
 *
 * Note: All operations use Ascend C compatible math
 * (same patterns as gaff2_force.h — no <math.h>, no __builtin_*).
 */

#ifndef PME_SPREAD_H
#define PME_SPREAD_H

#ifdef PME_DEVICE
#include "kernel_operator.h"
#include "pme_types.h"
using namespace AscendC;
#endif

// ============================================================
// B-spline values and derivatives (cubic, order 4)
//
// These compute the cubic B-spline value and derivative
// for a given fractional coordinate offset u ∈ [-2, 2).
//
// The cubic B-spline is non-zero over 4 grid points.
// ============================================================

// --- Reciprocal (1/x) via Newton-Raphson ---
__aicore__ inline float pme_recipf(float x) {
    if (x == 0.0f) return 0.0f;
    int32_t i = *(int32_t*)&x;
    i = 0x7EEEEEEE - i;
    float y = *(float*)&i;
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}

// --- Approximate exponent: exp(-x) for x >= 0 ---
__aicore__ inline float pme_expf_neg(float x) {
    if (x <= 0.0f) return 1.0f;
    if (x > 88.0f) return 0.0f;
    const float inv_ln2 = 1.4426950408889634f;
    float nf = x * inv_ln2;
    int32_t n = (int32_t)(nf + 0.5f);
    float r = nf - (float)n;
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

/**
 * Cubic B-spline (order 4) — Cox-de Boor formula
 *
 * Standard PME B-spline M_n(x) with support [-n/2, n/2].
 * For n=4 (cubic): support [-2, 2].
 *
 * M_n(x) = [1/(n-1)!] · Σ_{k=0}^n (-1)^k · C(n,k) · (x + n/2 - k)^{n-1}_+
 *
 * Closed form for cubic (n=4):
 *   M_4(x) =
 *     (2/3) - x² + |x|³/2               for |x| ≤ 1
 *     (2 - |x|)³ / 6                     for 1 < |x| < 2
 *     0                                   otherwise
 *
 * Note: This is the symmetric, centered form used in PME,
 * NOT the recursive θ_n form with support [0, n).
 */
__aicore__ inline float pme_bspline_cubic(float x) {
    float abs_x = x;
    if (x < 0.0f) abs_x = -x;

    if (abs_x >= 2.0f) return 0.0f;

    if (abs_x <= 1.0f) {
        // (2/3) - x² + |x|³/2
        float x2 = abs_x * abs_x;
        return 0.6666666666666666f - x2 + 0.5f * x2 * abs_x;
    } else {
        // (2 - |x|)³ / 6
        float t = 2.0f - abs_x;
        return 0.16666666666666666f * t * t * t;
    }
}

/**
 * Cubic B-spline derivative (n=4)
 *
 * dM_4/dx:
 *   -2x + 1.5·|x|·x               for |x| ≤ 1   (= -2x + 1.5·x·|x|)
 *   -(2 - |x|)² / 2 · sign(x)     for 1 < |x| < 2
 *   0                               otherwise
 */
__aicore__ inline float pme_bspline_deriv_cubic(float x) {
    float abs_x = x;
    if (x < 0.0f) abs_x = -x;

    if (abs_x >= 2.0f) return 0.0f;

    if (abs_x <= 1.0f) {
        // d/dx: -2x + 1.5·|x|·x
        return -2.0f * x + 1.5f * abs_x * x;
    } else {
        // d/dx: -(2 - |x|)² / 2 · sign(x)
        float t = 2.0f - abs_x;
        float val = -0.5f * t * t;
        return (x >= 0.0f) ? val : -val;
    }
}

// ============================================================
// PME Charge Spreading Kernel
//
// For each atom i:
//   u_i = (r_i / box_size) * mesh_dim   (fractional coords)
//   m_i = floor(u_i - (spline_order/2)) (starting grid point)
//   For each offset d in [0, spline_order):
//     Q[m_i_x + d_x][m_i_y + d_y][m_i_z + d_z] += q_i * θ(u_x - m_i_x) * θ(u_y - m_i_y) * θ(u_z - m_i_z)
//
// Parallelization: Each atom is assigned to one logical block.
// Since charge spreading uses atomic-style additive updates to mesh,
// we need to handle conflicts. Strategy: each block works on a
// non-overlapping subset of atoms. For M³ << N, multiple atoms
// may update the same grid point → use per-block local accumulators
// or serialize with one block.
//
// For simplicity: SINGLE BLOCK (blockDim=1) processes all atoms.
// M=32 → 32768 grid points. Each atom updates 4³=64 points.
// N=1000 atoms → 64000 updates, all sequential, no conflicts.
//
// For large systems (N >> M³): multi-block with hash-based
// grid partitioning.
// ============================================================

class PMESpreadKernel {
public:
    __aicore__ PMESpreadKernel() {}
    __aicore__ ~PMESpreadKernel() {}

    /**
     * Spread all atom charges onto the mesh grid.
     *
     * @param config  PME configuration
     * @param gm      Global memory pointers
     * @param start   First atom index (for this block)
     * @param end     One-past-last atom index
     */
    __aicore__ void SpreadCharges(
        PMEConfig config,
        PMEGlobalMemory gm,
        int32_t start,
        int32_t end)
    {
        // Decode GM pointers
        __gm__ float* coords;
        __gm__ float* charges;
        __gm__ float* mesh_re;

        __builtin_memcpy(&coords, &gm.coords_gm, sizeof(coords));
        __builtin_memcpy(&charges, &gm.charges_gm, sizeof(charges));
        __builtin_memcpy(&mesh_re, &gm.mesh_re_gm, sizeof(mesh_re));

        int32_t n_atoms = config.n_atoms;
        int32_t M = config.mesh_dim;
        int32_t order = config.spline_order;
        float inv_box = config.inv_box;
        float Mf = (float)M;

        // Clear mesh to zero
        for (int32_t i = 0; i < M * M * M; i++) {
            mesh_re[i] = 0.0f;
        }

        // For each atom in our range
        int32_t atom_end = (end > n_atoms) ? n_atoms : end;
        for (int32_t ai = start; ai < atom_end; ai++) {
            float xi = coords[ai * 3 + 0];
            float yi = coords[ai * 3 + 1];
            float zi = coords[ai * 3 + 2];
            float qi = charges[ai];

            // Fractional coordinates in grid units: u = r/L * M
            float ux = xi * inv_box * Mf;
            float uy = yi * inv_box * Mf;
            float uz = zi * inv_box * Mf;

            // Grid point just below u: m0 = floor(u)
            // For cubic B-spline with support [-2, 2], contributing grid points
            // are m = m0-1, m0, m0+1, m0+2
            // B-spline value: M_n(u - m) = M_n(u - m)
            int32_t mx0 = (int32_t)ux;  // floor(ux) for ux >= 0
            int32_t my0 = (int32_t)uy;
            int32_t mz0 = (int32_t)uz;

            // Precompute spline values for each offset (d = -1, 0, 1, 2)
            // Store in array indexed by [d+1]
            float bx[4], by[4], bz[4];
            for (int32_t d = -1; d <= 2; d++) {
                int32_t mi = mx0 + d;
                bx[d + 1] = pme_bspline_cubic(ux - (float)mi);
            }
            for (int32_t d = -1; d <= 2; d++) {
                int32_t mi = my0 + d;
                by[d + 1] = pme_bspline_cubic(uy - (float)mi);
            }
            for (int32_t d = -1; d <= 2; d++) {
                int32_t mi = mz0 + d;
                bz[d + 1] = pme_bspline_cubic(uz - (float)mi);
            }

            // Spread to 4×4×4 = 64 grid points
            for (int32_t dz = -1; dz <= 2; dz++) {
                int32_t mz = mz0 + dz;
                // PBC wrapping
                mz = (mz % M + M) % M;
                float b_z = bz[dz + 1];

                for (int32_t dy = -1; dy <= 2; dy++) {
                    int32_t my = my0 + dy;
                    my = (my % M + M) % M;
                    float b_zy = b_z * by[dy + 1];

                    for (int32_t dx = -1; dx <= 2; dx++) {
                        int32_t mx = mx0 + dx;
                        mx = (mx % M + M) % M;
                        float b_zyx = b_zy * bx[dx + 1];

                        // mesh[mz][my][mx] += q_i * b(x) * b(y) * b(z)
                        int32_t idx = (mz * M + my) * M + mx;
                        mesh_re[idx] += qi * b_zyx;
                    }
                }
            }
        }
    }

    /**
     * Interpolate reciprocal-space forces from grid back to atoms.
     *
     * For each atom i:
     *   F_i = q_i · Σ_{d} θ'_x · θ_y · θ_z · F_recon(m_i + d)
     *
     * where F_recon is the IFFT result read from mesh_gm.
     *
     * The forces are ADDED to the existing forces in forces_gm.
     *
     * @param config  PME configuration
     * @param gm      Global memory pointers
     * @param start   First atom index
     * @param end     One-past-last
     */
    __aicore__ void InterpolateForces(
        PMEConfig config,
        PMEGlobalMemory gm,
        int32_t start,
        int32_t end)
    {
        // Decode GM pointers
        __gm__ float* coords;
        __gm__ float* charges;
        __gm__ float* forces;
        __gm__ float* mesh_re;  // IFFT result (grid forces)
        __gm__ float* mesh_im;  // should be zero for real IFFT

        __builtin_memcpy(&coords, &gm.coords_gm, sizeof(coords));
        __builtin_memcpy(&charges, &gm.charges_gm, sizeof(charges));
        __builtin_memcpy(&forces, &gm.forces_gm, sizeof(forces));
        __builtin_memcpy(&mesh_re, &gm.mesh_re_gm, sizeof(mesh_re));
        __builtin_memcpy(&mesh_im, &gm.mesh_im_gm, sizeof(mesh_im));

        int32_t M = config.mesh_dim;
        float inv_box = config.inv_box;
        float Mf = (float)M;

        int32_t atom_end = (end > config.n_atoms) ? config.n_atoms : end;

        for (int32_t ai = start; ai < atom_end; ai++) {
            float xi = coords[ai * 3 + 0];
            float yi = coords[ai * 3 + 1];
            float zi = coords[ai * 3 + 2];
            float qi = charges[ai];

            // Fractional coordinates in grid units
            float ux = xi * inv_box * Mf;
            float uy = yi * inv_box * Mf;
            float uz = zi * inv_box * Mf;

            int32_t mx0 = (int32_t)ux;  // floor(ux)
            int32_t my0 = (int32_t)uy;
            int32_t mz0 = (int32_t)uz;

            // Precompute B-spline values AND derivatives for this atom
            // Using d = -1..2, store at [d+1]
            float bx[4], by[4], bz[4];
            float dbx[4], dby[4], dbz[4];

            for (int32_t d = -1; d <= 2; d++) {
                int32_t mi = mx0 + d;
                float u_diff = ux - (float)mi;
                bx[d + 1] = pme_bspline_cubic(u_diff);
                dbx[d + 1] = pme_bspline_deriv_cubic(u_diff);
            }
            for (int32_t d = -1; d <= 2; d++) {
                int32_t mi = my0 + d;
                float u_diff = uy - (float)mi;
                by[d + 1] = pme_bspline_cubic(u_diff);
                dby[d + 1] = pme_bspline_deriv_cubic(u_diff);
            }
            for (int32_t d = -1; d <= 2; d++) {
                int32_t mi = mz0 + d;
                float u_diff = uz - (float)mi;
                bz[d + 1] = pme_bspline_cubic(u_diff);
                dbz[d + 1] = pme_bspline_deriv_cubic(u_diff);
            }

            // Compute forces via B-spline gradient interpolation
            // F_i = q_i * (inv_box * M) * Σ_{mesh} [θ'_x·θ_y·θ_z·F_recon]
            float grad_scale = inv_box * Mf;

            float fx = 0.0f, fy = 0.0f, fz = 0.0f;

            for (int32_t dz = -1; dz <= 2; dz++) {
                int32_t mz = (mz0 + dz) % M;
                if (mz < 0) mz += M;

                for (int32_t dy = -1; dy <= 2; dy++) {
                    int32_t my = (my0 + dy) % M;
                    if (my < 0) my += M;

                    for (int32_t dx = -1; dx <= 2; dx++) {
                        int32_t mx = (mx0 + dx) % M;
                        if (mx < 0) mx += M;

                        int32_t idx = (mz * M + my) * M + mx;
                        float f_mesh = mesh_re[idx];  // IFFT result (force density)

                        // B-spline gradient formula:
                        // F_i_x += q_i * dbx[dx+1] * by[dy+1] * bz[dz+1] * f_mesh * scale
                        fx += dbx[dx + 1] * by[dy + 1] * bz[dz + 1] * f_mesh;
                        fy += bx[dx + 1] * dby[dy + 1] * bz[dz + 1] * f_mesh;
                        fz += bx[dx + 1] * by[dy + 1] * dbz[dz + 1] * f_mesh;
                    }
                }
            }

            forces[ai * 3 + 0] += qi * fx * grad_scale;
            forces[ai * 3 + 1] += qi * fy * grad_scale;
            forces[ai * 3 + 2] += qi * fz * grad_scale;
        }
    }
};

#endif // PME_SPREAD_H
