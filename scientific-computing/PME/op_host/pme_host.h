/**
 * pme_host.h
 *
 * Particle Mesh Ewald (PME) — Host-Side Electrostatic Solver
 *
 * Implements a complete Ewald summation with PME reciprocal-space
 * computation via 3D FFT (fftw3f). Replaces the simple cutoff Coulomb
 * in the NPU kernel with accurate long-range electrostatics.
 *
 * Architecture (called from GAFF2ForceEvaluator):
 *   1. Compute NPU kernel (LJ + cutoff Coulomb 1/r)
 *   2. Subtract cutoff Coulomb 1/r contribution from forces+energy
 *   3. Add Ewald sum: direct(erfc/r) + reciprocal(FFT) + self
 *
 * Units: kJ/mol, nm, e
 *
 * Reference: Essmann et al., JCP 103, 8577 (1995)
 */

#ifndef PME_HOST_H
#define PME_HOST_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <complex>
#include <fftw3.h>

static constexpr double COULOMB_C = 138.935458;  // kJ·nm·mol⁻¹·e⁻²

// ============================================================
// PME Configuration
// ============================================================
struct PMEConfig {
    int32_t n_atoms = 0;
    double box_size = 10.0;       // nm
    double ewald_alpha = 2.5;     // nm⁻¹
    int32_t mesh_dim = 32;        // FFT grid
    int32_t spline_order = 4;     // B-spline order
    double direct_cutoff = 1.0;   // nm
};

// ============================================================
// PME Solver
// ============================================================
class PMESolver {
public:
    PMESolver();
    ~PMESolver() { Finalize(); }

    // Initialize
    int32_t Initialize(const PMEConfig& config);

    // Set charges
    void SetCharges(const double* charges, int32_t n_atoms);

    // Set coordinates
    void SetCoordinates(const double* coords, int32_t n_atoms);

    // Main solve: compute reciprocal-space forces and energy
    // forces_recip is ADDED to in-place
    void ComputeForces(double* forces, double* potential);

    // Compute all Ewald components and return breakdown
    struct EwaldBreakdown {
        double direct_energy;
        double recip_energy;
        double self_energy;
        double correction_energy;  // (direct+recip+self) - cutoff_coulomb
    };
    EwaldBreakdown GetBreakdown() const;

    // Cleanup
    void Finalize();

    bool IsInitialized() const { return initialized_; }

private:
    // B-spline values and derivatives
    void bspline(double u, int order, double* theta, double* dtheta);

    // Charge spreading onto grid
    void spread_charges();

    // Compute influence function B(m)
    void compute_influence_function();

    // Reciprocal convolution (multiply Q~ by B)
    void reciprocal_convolution(double* forces);

    // Direct-space Ewald sum (erfc/r)
    void compute_direct_ewald(const double* coords, double* forces, double& energy);

    // Self energy
    double compute_self_energy() const;

    // Subtract cutoff Coulomb (1/r) from forces + energy
    void subtract_cutoff_coulomb(const double* coords, double* forces, double& energy);

    // Approximate erfc(x) for x >= 0
    static double erfcf_approx(double x);

    // Helper: minimum image displacement
    static void min_image(double dx, double dy, double dz, double half_box, double box_size,
                          double& odx, double& ody, double& odz);

    bool initialized_;
    PMEConfig cfg_;

    int32_t mesh_dim_;
    int32_t spline_order_;
    int32_t n_atoms_;
    double box_size_, half_box_;
    double alpha_, alpha_sq_;
    double volume_;
    double cutoff_, cutoff_sq_;
    double inv_box_;

    // Input buffers
    std::vector<double> charges_;
    std::vector<double> coords_;

    // Fractional coordinates
    std::vector<double> frac_;

    // FFT grids
    int32_t grid_size_;
    fftwf_complex* Q_;      // complex grid for FFT
    fftwf_complex* Qtilde_; // Fourier space grid
    double* B_;             // influence function

    // FFTW plans
    fftwf_plan plan_fwd_;
    fftwf_plan plan_bwd_;

    // Results
    double direct_energy_;
    double recip_energy_;
    double self_energy_;
    double coulomb_cutoff_energy_; // original 1/r Coulomb from cutoff

    // Scratch buffers
    std::vector<double> bsp_coeff_; // spline coeffs for current atom
};

#endif // PME_HOST_H
