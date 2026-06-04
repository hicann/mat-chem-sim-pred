/**
 * vv_host.h
 *
 * VV Integrator + NPT Thermostat/Barostat — Host-Side API
 *
 * Provides AscendCL interface for launching the VV and thermo/barostat
 * kernels on Ascend NPU. The host handles:
 *   1. Kernel registration (RegisterAscendBinary) for 3 kernels
 *   2. Launch orchestration: vv_integrate → force_eval → vv_finish → thermo_scale
 *   3. Host-side scalar computations (KE reduction, λ/μ calculation)
 *   4. RNG for Andersen and V-rescale (C++ std RNG on host)
 *
 * Usage:
 *   VVHost vv;
 *   vv.Initialize(n_atoms, dt, masses, n_dof, device_id);
 *   vv.Step(coords_dev, velocities_dev, forces_dev, box_size);
 *     → returns KE, temperature, virial (host-side values)
 *
 * Memory layout (device GM buffers, shared with GAFF2/Evald):
 *   coords_gm:     float[n_atoms][3]
 *   velocities_gm: float[n_atoms][3]  (NOT shared with GAFF2)
 *   forces_gm:     float[n_atoms][3]  (shared with GAFF2)
 *   masses_gm:     float[n_atoms]
 *   pot_virial_gm: float[4]
 */

#ifndef VV_HOST_H
#define VV_HOST_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "vv_types.h"

// Boltzmann constant in kJ/(mol·K)
static constexpr double KB_VVH = 8.314462618e-3;

class VVHost {
public:
    VVHost()
        : initialized_(false), n_atoms_(0), device_id_(0),
          context_(nullptr), stream_(nullptr),
          coords_(nullptr), velocities_(nullptr), forces_(nullptr),
          masses_(nullptr), pot_virial_(nullptr),
          pot_virial_host_(nullptr),
          rng_(42), uniform_dist_(0.0, 1.0) {}

    ~VVHost() { Finalize(); }

    // ============================================================
    // Initialize
    //   n_atoms  — number of atoms
    //   dt       — time step (ps)
    //   masses   — atomic masses (g/mol), or nullptr for all 1.0
    //   n_dof    — degrees of freedom (3N - 3 for COM removal)
    //   device_id — NPU device ID (default: 0)
    //
    //   shared_coords, shared_forces — GM buffers from GAFF2Host
    //   (velocities_ and pot_virial_ are allocated internally)
    // ============================================================
    int32_t Initialize(
        int32_t n_atoms, double dt,
        const double* masses,
        int32_t n_dof,
        void* shared_coords,
        void* shared_forces,
        aclrtContext external_context = nullptr,
        aclrtStream external_stream = nullptr,
        int32_t device_id = 0);

    // ============================================================
    // PERFORM ONE FULL NPT STEP (all 3 kernels + host scalar calc)
    //
    // Caller must have already ensured forces_gm contains forces at
    // the current coordinates.
    //
    // Returns: KE (kinetic energy in kJ/mol)
    // Outputs: temperature, virial (host-side scalars)
    //          box_size (updated by barostat, if enabled)
    // ============================================================
    double Step(
        double* temperature_out,   // [out] instantaneous T
        double* virial_out,         // [out] virial trace
        double* box_size,           // [in/out] current box size
        double target_temp,         // K
        double target_pressure,     // kJ/(mol·nm³)
        double tau_t,               // ps, V-rescale coupling
        double tau_p,               // ps, C-rescale coupling
        double collision_freq,      // ps⁻¹, Andersen freq
        double compressibility      // (kJ/(mol·nm³))⁻¹
    );

    // ============================================================
    // SPLIT NPU KERNEL LAUNCHES (for NPTIntegrator integration)
    //
    // These allow precise control of launch order:
    //   StepIntegrate()  — vv_integrate (v_half + position update)
    //   → [host: force evaluation at new coords]
    //   StepFinish()     — vv_finish (v_finish + KE/virial) + host scalars
    //   StepScale()      — thermo_scale (V-rescale + C-rescale scaling)
    //
    // Parameters set via the respective Set... methods.
    // ============================================================
    
    // Set NPT parameters (must be called before StepFinish/StepScale)
    void SetNPTParams(
        double dt, double target_temp, double target_pressure,
        double tau_t, double tau_p, double collision_freq,
        double compressibility)
    {
        dt_ = dt;
        half_dt_ = dt * 0.5;
        cfg_target_temp_ = target_temp;
        cfg_target_pressure_ = target_pressure;
        cfg_tau_t_ = tau_t;
        cfg_tau_p_ = tau_p;
        cfg_collision_freq_ = collision_freq;
        cfg_compressibility_ = compressibility;
    }

    // Step 1: vv_integrate kernel only (v_half + position update)
    // Requires: forces on device from previous force eval
    int32_t StepIntegrate();

    // Step 2a: vv_finish kernel + host-side scalar calculations
    // Requires: forces on device from force eval at new coords
    // Returns: KE, outputs: temperature, virial, lambda, mu, v_scale
    void StepFinish(
        double* temperature_out,
        double* virial_out,
        double* box_size);

    // Step 2b: Apply V-rescale lambda from StepFinish
    // (only if V-rescale was enabled)
    int32_t StepVRescale();

    // Step 3: thermo_scale kernel (applies V-rescale and/or C-rescale)
    // Uses: vrescale_lambda_ and crescale_mu_/v_scale_ from StepFinish
    int32_t StepScale();

    // Access computed scalars from StepFinish
    double GetComputedLambda() const { return vrescale_lambda_; }
    double GetComputedMu()     const { return crescale_mu_; }
    double GetComputedVScale() const { return v_scale_; }

    // ============================================================
    // Access device pointers
    // ============================================================
    void* GetCoordinatesDevicePtr()   const { return coords_; }
    void* GetVelocitiesDevicePtr()    const { return velocities_; }
    void* GetForcesDevicePtr()        const { return forces_; }

    aclrtContext GetContext() const { return context_; }
    aclrtStream  GetStream()  const { return stream_; }

    bool IsInitialized() const { return initialized_; }
    int32_t GetNAtoms() const { return n_atoms_; }

    // ============================================================
    // Cleanup
    // ============================================================
    void Finalize();

private:
    // ============================================================
    // Kernel launch helpers
    // ============================================================
    int32_t LaunchVVIntegrate(VVConfig& config);
    int32_t LaunchVVFinish(VVConfig& config);
    int32_t LaunchThermoScale(VVConfig& config);

    // ============================================================
    // Host-side scalar computation helpers
    // ============================================================
    double ComputeKEFromDevice();
    double ComputeVirialFromDevice();
    double ComputePressure(double ke, double virial, double volume) const {
        return (2.0 * ke + virial) / (3.0 * volume);
    }
    double BoxMuller();

    bool initialized_;
    int32_t n_atoms_;
    int32_t device_id_;
    int32_t n_dof_;
    double dt_;
    double half_dt_;

    aclrtContext context_;
    aclrtStream stream_;

    // Device memory
    void* coords_;         // shared from GAFF2Host — DO NOT free
    void* velocities_;     // owned — float[n_atoms][3]
    void* forces_;         // shared from GAFF2Host — DO NOT free
    void* masses_;         // owned — float[n_atoms]
    void* pot_virial_;     // owned — float[4]

    // Host memory
    float* pot_virial_host_;

    // RNG
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    double andersen_kT_;   // kT for Andersen (precomputed)
    bool andersen_enabled_;

    // Mass cache
    std::vector<float> masses_float_;

    // Cached config values (for split step API)
    double cfg_target_temp_ = 300.0;
    double cfg_target_pressure_ = 0.0;
    double cfg_tau_t_ = 0.0;
    double cfg_tau_p_ = 0.0;
    double cfg_collision_freq_ = 0.0;
    double cfg_compressibility_ = 0.5;

    // Computed scalars (from StepFinish, used by StepScale)
    double vrescale_lambda_ = 1.0;
    double crescale_mu_ = 1.0;
    double v_scale_ = 1.0;
};

#endif // VV_HOST_H
