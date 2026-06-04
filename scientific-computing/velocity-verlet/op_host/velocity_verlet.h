/**
 * velocity_verlet.h
 *
 * Velocity Verlet Integrator — Host-Side CPU Implementation
 *
 * Algorithm (one full step):
 *   1. v(t+dt/2) = v(t) + (dt/2) * a(t)           [half-step velocity]
 *   2. r(t+dt)   = r(t) + dt * v(t+dt/2)           [position update]
 *   3. Compute forces at r(t+dt)                    [via ForceEvaluator]
 *   4. v(t+dt)   = v(t+dt/2) + (dt/2) * a(t+dt)    [half-step velocity]
 *
 * All units:
 *   time:  ps
 *   coords: nm
 *   velocities: nm/ps
 *   forces: kJ/(mol·nm)
 *   mass: g/mol (=> 1.0 for non-hydrogen atoms)
 *
 * Note: Since masses are in g/mol, the acceleration a = F/m has
 * correct dimensions because kJ/(mol·nm) / (g/mol) = kJ/(g·nm)
 * which works out with the time unit (ps) for the conversion factor.
 */

#ifndef VELOCITY_VERLET_H
#define VELOCITY_VERLET_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#include "force_evaluator.h"

// Boltzmann constant in kJ/(mol·K)
static constexpr double KB_VV = 8.314462618e-3;

class VVIntegrator {
public:
    VVIntegrator()
        : evaluator_(nullptr), n_atoms_(0), dt_(0.001),
          uniform_mass_(true), uniform_mass_value_(1.0),
          initialized_(false) {}

    ~VVIntegrator() { Finalize(); }

    // ============================================================
    // Initialize
    // ============================================================
    int32_t Initialize(
        ForceEvaluator* evaluator,
        int32_t n_atoms,
        double dt,              // ps
        const double* masses = nullptr  // g/mol, nullptr → all 1.0
    ) {
        if (!evaluator) {
            fprintf(stderr, "[VV] ERROR: ForceEvaluator is null\n");
            return -1;
        }

        evaluator_ = evaluator;
        n_atoms_ = n_atoms;
        dt_ = dt;

        // Set up masses
        atom_masses_.resize(n_atoms_);
        if (masses) {
            uniform_mass_ = true;
            uniform_mass_value_ = masses[0];
            for (int32_t i = 0; i < n_atoms_; i++) {
                atom_masses_[i] = masses[i];
                if (std::abs(masses[i] - uniform_mass_value_) > 1e-10) {
                    uniform_mass_ = false;
                }
            }
        } else {
            uniform_mass_ = true;
            uniform_mass_value_ = 1.0;
            for (int32_t i = 0; i < n_atoms_; i++) {
                atom_masses_[i] = 1.0;
            }
        }

        initialized_ = true;
        return 0;
    }

    // ============================================================
    // Initialize velocities from Maxwell-Boltzmann distribution
    // ============================================================
    void InitVelocities(
        double* velocities,
        double temperature,
        uint64_t seed = 42
    ) {
        if (!initialized_) return;

        // Box-Muller transform for Gaussian random velocities
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        double kT = KB_VV * temperature;
        double total_momentum[3] = {0, 0, 0};

        for (int32_t i = 0; i < n_atoms_; i++) {
            double mass = atom_masses_[i];
            double sigma = std::sqrt(kT / mass);

            for (int32_t d = 0; d < 3; d++) {
                double u1 = uniform(rng);
                double u2 = uniform(rng);
                double z = std::sqrt(-2.0 * std::log(u1 + 1e-30)) * std::cos(2.0 * M_PI * u2);
                velocities[i * 3 + d] = z * sigma;
                total_momentum[d] += mass * velocities[i * 3 + d];
            }
        }

        // Remove center-of-mass momentum
        double total_mass = 0;
        for (int32_t i = 0; i < n_atoms_; i++) {
            total_mass += atom_masses_[i];
        }
        for (int32_t d = 0; d < 3; d++) {
            total_momentum[d] /= total_mass;
        }
        for (int32_t i = 0; i < n_atoms_; i++) {
            for (int32_t d = 0; d < 3; d++) {
                velocities[i * 3 + d] -= total_momentum[d];
            }
        }
    }

    // ============================================================
    // Perform one full Velocity Verlet step
    //
    // Input:  coords, velocities at time t
    // Input:  forces at time t (must be pre-computed at coords[t])
    // Output: coords[t+dt], velocities[t+dt]
    // Output: forces at coords[t+dt] (evaluated by ForceEvaluator)
    // ============================================================
    aclError Step(
        double* coords,
        double* velocities,
        double* forces,
        double* potential_out = nullptr
    ) {
        if (!initialized_) return ACL_ERROR_FAILURE;

        double half_dt = dt_ * 0.5;

        // Step 1: v(t+dt/2) = v(t) + (dt/2) * F(t) / m
        for (int32_t i = 0; i < n_atoms_; i++) {
            double inv_m = 1.0 / atom_masses_[i];
            for (int32_t d = 0; d < 3; d++) {
                velocities[i * 3 + d] += half_dt * forces[i * 3 + d] * inv_m;
            }
        }

        // Step 2: r(t+dt) = r(t) + dt * v(t+dt/2)
        for (int32_t i = 0; i < n_atoms_; i++) {
            for (int32_t d = 0; d < 3; d++) {
                coords[i * 3 + d] += dt_ * velocities[i * 3 + d];
            }
        }

        // Step 3: Evaluate forces at new coordinates
        double potential = 0.0;
        aclError ret = evaluator_->EvaluateForces(coords, forces, &potential);
        if (ret != ACL_SUCCESS) return ret;

        // Step 4: v(t+dt) = v(t+dt/2) + (dt/2) * F(t+dt) / m
        for (int32_t i = 0; i < n_atoms_; i++) {
            double inv_m = 1.0 / atom_masses_[i];
            for (int32_t d = 0; d < 3; d++) {
                velocities[i * 3 + d] += half_dt * forces[i * 3 + d] * inv_m;
            }
        }

        if (potential_out) {
            *potential_out = potential;
        }

        return ACL_SUCCESS;
    }

    // ============================================================
    // Compute kinetic energy from velocities
    // ============================================================
    double ComputeKineticEnergy(const double* velocities) const {
        double ke = 0.0;
        for (int32_t i = 0; i < n_atoms_; i++) {
            double vx = velocities[i * 3 + 0];
            double vy = velocities[i * 3 + 1];
            double vz = velocities[i * 3 + 2];
            ke += atom_masses_[i] * (vx * vx + vy * vy + vz * vz);
        }
        return 0.5 * ke;
    }

    // ============================================================
    // Compute instantaneous temperature from velocities
    // T = 2 * KE / (n_dof * kB)
    // ============================================================
    double ComputeTemperature(const double* velocities) const {
        double ke = ComputeKineticEnergy(velocities);
        int32_t n_dof = 3 * n_atoms_ - 3;  // -3 for COM removal
        return 2.0 * ke / (n_dof * KB_VV);
    }

    void Finalize() {
        evaluator_ = nullptr;
        initialized_ = false;
    }

    bool IsInitialized() const { return initialized_; }
    double GetDT() const { return dt_; }
    int32_t GetNAtoms() const { return n_atoms_; }

private:
    ForceEvaluator* evaluator_;
    int32_t n_atoms_;
    double dt_;
    bool uniform_mass_;
    double uniform_mass_value_;
    std::vector<double> atom_masses_;
    bool initialized_;
};

#endif // VELOCITY_VERLET_H
