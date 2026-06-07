/**
 * shake.h
 *
 * SHAKE Algorithm for Bond Length Constraints — CPU Host Implementation
 *
 * SHAKE is an iterative constraint solver that adjusts atomic positions
 * to maintain fixed bond lengths. It is applied after the VV position
 * update, before force evaluation.
 *
 * Algorithm:
 *   For each constrained bond (i,j) with target length d₀:
 *     λ = [r²(t+dt) - d₀²] / [2·dt²·(1/m_i + 1/m_j) · r(t+dt) · (r_i - r_j)]
 *     r_i ← r_i + λ·dt²/m_i · (r_i - r_j)
 *     r_j ← r_j - λ·dt²/m_j · (r_i - r_j)
 *
 *   Iterate until all bonds satisfy |r² - d₀²| < tolerance
 *
 * Units:
 *   coords: nm
 *   bond lengths: nm
 *   masses: g/mol
 *   dt: ps
 *
 * Reference:
 *   Ryckaert, Ciccotti, Berendsen, JCP 23, 327 (1977)
 *   "Numerical integration of the Cartesian equations of motion of a
 *    system with constraints: molecular dynamics of n-alkanes"
 */

#ifndef SHAKE_H
#define SHAKE_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

// ============================================================
// SHAKE Configuration
// ============================================================
struct SHAKEConfig {
    int32_t n_atoms = 0;          // Number of atoms
    int32_t n_constraints = 0;    // Number of constrained bonds
    const int32_t* constraints = nullptr;  // [n_constraints][2] — 1-indexed atom pairs
    const double* target_lengths = nullptr; // [n_constraints] — equilibrium bond lengths (nm)
    const double* masses = nullptr;         // [n_atoms] — atomic masses (g/mol)
    double dt = 0.001;             // Timestep (ps)
    double tolerance = 1e-10;      // Convergence tolerance (nm²)
    int32_t max_iterations = 100;  // Max SHAKE iterations
    bool verbose = false;
};

// ============================================================
// SHAKE State
// ============================================================
struct SHAKEState {
    int32_t iterations = 0;        // Last iteration count
    double max_error = 0.0;        // Last max constraint error (nm²)
    bool converged = false;        // Whether last call converged
    double rms_error = 0.0;        // RMS constraint error

    void Reset() {
        iterations = 0;
        max_error = 0.0;
        converged = false;
        rms_error = 0.0;
    }
};

// ============================================================
// SHAKE Solver Class
// ============================================================
class SHAKESolver {
public:
    SHAKESolver() : configured_(false), n_constraints_(0), n_atoms_(0) {}

    ~SHAKESolver() { Finalize(); }

    // ============================================================
    // Configure SHAKE constraints
    // ============================================================
    int32_t Configure(const SHAKEConfig& config) {
        if (config.n_atoms <= 0 || config.n_constraints <= 0) {
            fprintf(stderr, "[SHAKE] ERROR: invalid n_atoms or n_constraints\n");
            return -1;
        }
        if (!config.constraints || !config.target_lengths) {
            fprintf(stderr, "[SHAKE] ERROR: constraints or target_lengths is null\n");
            return -1;
        }

        cfg_ = config;
        n_atoms_ = config.n_atoms;
        n_constraints_ = config.n_constraints;

        // Copy constraints (convert from 1-indexed to 0-indexed internally)
        constraints_.resize(n_constraints_ * 2);
        target_lengths_.resize(n_constraints_);
        for (int32_t i = 0; i < n_constraints_; i++) {
            constraints_[i * 2 + 0] = config.constraints[i * 2 + 0] - 1;  // 0-indexed
            constraints_[i * 2 + 1] = config.constraints[i * 2 + 1] - 1;  // 0-indexed
            target_lengths_[i] = config.target_lengths[i];
        }

        // Precompute inverse masses
        inv_masses_.resize(n_atoms_);
        if (config.masses) {
            for (int32_t i = 0; i < n_atoms_; i++) {
                inv_masses_[i] = 1.0 / config.masses[i];
            }
        } else {
            for (int32_t i = 0; i < n_atoms_; i++) {
                inv_masses_[i] = 1.0;
            }
        }

        // Precompute dt² terms for each constraint
        dt_sq_ = config.dt * config.dt;
        inv_dt_sq_ = 1.0 / dt_sq_;

        configured_ = true;

        if (config.verbose) {
            fprintf(stdout, "[SHAKE] Configured: %d constraints on %d atoms\n",
                    n_constraints_, n_atoms_);
            fprintf(stdout, "[SHAKE] dt=%.4f, tol=%.2e, max_iter=%d\n",
                    config.dt, config.tolerance, config.max_iterations);
        }

        return 0;
    }

    // ============================================================
    // Apply SHAKE to positions after VV update
    //
    // Input:  positions from VV update (unconstrained)
    // Output: positions satisfying all bond constraints
    //
    // The algorithm iterates over all constraints until convergence.
    // ============================================================
    int32_t Apply(double* positions) {
        if (!configured_) {
            fprintf(stderr, "[SHAKE] ERROR: Not configured\n");
            return -1;
        }

        state_.Reset();
        double tolerance = cfg_.tolerance;
        int32_t max_iter = cfg_.max_iterations;

        for (int32_t iter = 0; iter < max_iter; iter++) {
            double max_err = 0.0;
            double sum_sq_err = 0.0;

            // Iterate over all constraints
            for (int32_t c = 0; c < n_constraints_; c++) {
                int32_t i = constraints_[c * 2 + 0];
                int32_t j = constraints_[c * 2 + 1];
                double d0_sq = target_lengths_[c] * target_lengths_[c];

                // Current bond vector
                double dx = positions[i * 3 + 0] - positions[j * 3 + 0];
                double dy = positions[i * 3 + 1] - positions[j * 3 + 1];
                double dz = positions[i * 3 + 2] - positions[j * 3 + 2];
                double r_sq = dx * dx + dy * dy + dz * dz;

                // Constraint error
                double error = r_sq - d0_sq;
                double abs_err = fabs(error);
                if (abs_err > max_err) max_err = abs_err;
                sum_sq_err += error * error;

                if (abs_err < tolerance) continue;

                // SHAKE multiplier λ
                // λ = [r² - d₀²] / [2·dt²·(1/m_i + 1/m_j) · (r·r)]
                // where r·r = dx·dx + dy·dy + dz·dz = r_sq
                // Actually the formula uses: λ = [r² - d₀²] / [2·dt²·(1/m_i+1/m_j) · (r·r)]
                // But the standard SHAKE updates are:
                //   r_i += λ·dt²/m_i · (r_i - r_j)
                //   r_j -= λ·dt²/m_j · (r_i - r_j)
                // where λ = [r² - d₀²] / [2·dt²·(1/m_i+1/m_j) · r²]

                double inv_m_sum = inv_masses_[i] + inv_masses_[j];
                double denominator = 2.0 * dt_sq_ * inv_m_sum * r_sq;

                if (denominator < 1e-30) continue;

                double lambda = error / denominator;

                // Apply position corrections
                double corr_scale_i = lambda * dt_sq_ * inv_masses_[i];
                double corr_scale_j = lambda * dt_sq_ * inv_masses_[j];

                positions[i * 3 + 0] -= corr_scale_i * dx;
                positions[i * 3 + 1] -= corr_scale_i * dy;
                positions[i * 3 + 2] -= corr_scale_i * dz;

                positions[j * 3 + 0] += corr_scale_j * dx;
                positions[j * 3 + 1] += corr_scale_j * dy;
                positions[j * 3 + 2] += corr_scale_j * dz;
            }

            state_.max_error = max_err;
            state_.rms_error = sqrt(sum_sq_err / n_constraints_);
            state_.iterations = iter + 1;

            if (max_err < tolerance) {
                state_.converged = true;
                if (cfg_.verbose && iter > 5) {
                    fprintf(stdout, "[SHAKE] Converged: iter=%d, max_err=%.2e, rms=%.2e\n",
                            iter + 1, max_err, state_.rms_error);
                }
                return iter + 1;
            }

            // Divergence check
            if (iter > 5 && max_err > 1e-3) {
                // Might be diverging — allow but warn
                if (cfg_.verbose) {
                    fprintf(stdout, "[SHAKE] WARNING: slow convergence at iter %d: max_err=%.2e\n",
                            iter + 1, max_err);
                }
            }
        }

        // Did not converge
        state_.converged = false;
        fprintf(stderr, "[SHAKE] WARNING: Did not converge in %d iterations. max_err=%.2e\n",
                max_iter, state_.max_error);
        return -state_.iterations;  // Negative indicates non-convergence
    }

    // ============================================================
    // Compute constraint forces from SHAKE displacements
    //
    // The constraint force for bond (i,j) is:
    //   F_constraint_i = -m_i · Δr_i / dt²
    //
    // These can be added to the physical forces.
    // ============================================================
    /*
    void ComputeConstraintForces(
        const double* positions_before,  // positions before SHAKE
        const double* positions_after,   // positions after SHAKE
        double* constraint_forces         // [n_atoms * 3] output
    ) {
        memset(constraint_forces, 0, n_atoms_ * 3 * sizeof(double));
        for (int32_t c = 0; c < n_constraints_; c++) {
            int32_t i = constraints_[c * 2 + 0];
            int32_t j = constraints_[c * 2 + 1];

            // Displacement due to SHAKE
            double disp_i_x = positions_after[i*3+0] - positions_before[i*3+0];
            double disp_i_y = positions_after[i*3+1] - positions_before[i*3+1];
            double disp_i_z = positions_after[i*3+2] - positions_before[i*3+2];

            // Constraint force: F = -m * Δr / dt²
            double inv_dt2 = 1.0 / (cfg_.dt * cfg_.dt);
            constraint_forces[i*3+0] -= inv_masses_[i] * disp_i_x * inv_dt2;
            constraint_forces[i*3+1] -= inv_masses_[i] * disp_i_y * inv_dt2;
            constraint_forces[i*3+2] -= inv_masses_[i] * disp_i_z * inv_dt2;

            // For atom j, force is opposite
            double disp_j_x = positions_after[j*3+0] - positions_before[j*3+0];
            double disp_j_y = positions_after[j*3+1] - positions_before[j*3+1];
            double disp_j_z = positions_after[j*3+2] - positions_before[j*3+2];

            constraint_forces[j*3+0] -= inv_masses_[j] * disp_j_x * inv_dt2;
            constraint_forces[j*3+1] -= inv_masses_[j] * disp_j_y * inv_dt2;
            constraint_forces[j*3+2] -= inv_masses_[j] * disp_j_z * inv_dt2;
        }
    }
    */

    // ============================================================
    // Access results
    // ============================================================
    const SHAKEState& GetState() const { return state_; }
    bool WasConverged() const { return state_.converged; }
    int32_t GetIterations() const { return state_.iterations; }
    double GetMaxError() const { return state_.max_error; }
    double GetRMSError() const { return state_.rms_error; }

    void SetVerbose(bool v) { cfg_.verbose = v; }

    // ============================================================
    // Finalize
    // ============================================================
    void Finalize() {
        configured_ = false;
        constraints_.clear();
        target_lengths_.clear();
        inv_masses_.clear();
    }

    bool IsConfigured() const { return configured_; }

private:
    SHAKEConfig cfg_;
    bool configured_;
    int32_t n_atoms_;
    int32_t n_constraints_;
    SHAKEState state_;

    // Internal data
    std::vector<int32_t> constraints_;       // [n_constraints * 2] — 0-indexed pairs
    std::vector<double> target_lengths_;     // [n_constraints]
    std::vector<double> inv_masses_;         // [n_atoms]
    double dt_sq_;
    double inv_dt_sq_;
};

#endif // SHAKE_H
