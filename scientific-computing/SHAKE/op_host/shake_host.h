/**
 * shake_host.h
 *
 * SHAKE Host-Side API — NPU Kernel Launch + Convergence Loop
 *
 * Provides AscendCL interface for launching the SHAKE iteration kernel.
 * The host loops over iterations: launch kernel → read max_error → check convergence.
 *
 * Architecture:
 *   Host convergence loop (for iter in 0..max_iter):
 *     → SHAKE iteration kernel (serial over all constraints)
 *     → Read max_error from device scalar
 *     → Check |error| < tolerance
 *
 * Usage:
 *   ShakeHost shake;
 *   shake.Initialize(n_atoms, inv_masses, context, stream);
 *   shake.SetConstraints(n_constraints, constraints, target_lengths, dt);
 *   shake.Apply(coords);           // host buffer
 *   // or: shake.ApplyDevice(coords_device); // device buffer
 *
 * Units:
 *   coords: nm
 *   bond lengths: nm
 *   masses: g/mol
 *   dt: ps
 *
 * Reference:
 *   Ryckaert, Ciccotti, Berendsen, JCP 23, 327 (1977)
 */
#ifndef SHAKE_HOST_H
#define SHAKE_HOST_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

#include "acl/acl.h"
#include "shake_types.h"

// ============================================================
// ShakeHost — NPU SHAKE solver
// ============================================================
class ShakeHost {
public:
    ShakeHost();
    ~ShakeHost();

    // ============================================================
    // Initialize with shared ACL context and stream
    // ============================================================
    int32_t Initialize(
        int32_t n_atoms,
        const float* inv_masses,       // [n_atoms] — 1/mass (can be nullptr for all 1.0)
        aclrtContext shared_context,
        aclrtStream shared_stream,
        int32_t device_id = 0,
        float tolerance = 1e-10f,
        int32_t max_iterations = 100
    );

    // ============================================================
    // Set constraints and timestep
    // ============================================================
    int32_t SetConstraints(
        int32_t n_constraints,
        const int32_t* constraints,    // [n_constraints * 2] — 0-indexed atom pairs
        const float* target_lengths,   // [n_constraints] — equilibrium bond lengths (nm)
        float dt = 0.001f              // ps — timestep
    );

    // ============================================================
    // Apply SHAKE to positions (host-side float buffer)
    //
    // coords: [n_atoms * 3] float — positions before/after SHAKE
    // Returns: number of iterations used (positive = converged,
    //          negative = did not converge)
    // ============================================================
    int32_t Apply(float* coords_host);

    // ============================================================
    // Apply SHAKE to device positions directly
    //
    // coords_device: device-side float buffer, modified in-place
    // Returns: number of iterations used (negative = not converged)
    // ============================================================
    int32_t ApplyDevice(float* coords_device);

    // ============================================================
    // Access results
    // ============================================================
    bool WasConverged() const;
    int32_t GetIterations() const;
    double GetMaxError() const;
    float* GetCoordinatesDevicePtr();

    // ============================================================
    // Configuration
    // ============================================================
    void SetTolerance(float tol);
    void SetMaxIterations(int32_t max_iter);
    void SetTimestep(float dt);
    bool IsInitialized() const;

    // ============================================================
    // Finalize — release device memory and unregister kernel
    // ============================================================
    void Finalize();

private:
    // Internal methods
    int32_t AllocateDeviceMemory();
    int32_t ApplyDeviceInternal();
    int32_t LaunchIteration();

private:
    bool initialized_ = false;
    bool acl_initialized_ = false;
    int32_t n_atoms_ = 0;
    int32_t n_constraints_ = 0;
    int32_t max_constraints_allocated_ = 0;
    int32_t max_iterations_ = 100;
    float tolerance_ = 1e-6f;
    float dt_sq_ = 1e-6f;
    bool device_owns_coords_ = false;

    int32_t device_id_ = 0;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;

    // Device memory
    float* coords_device_ = nullptr;
    int32_t* constraints_device_ = nullptr;
    float* target_lengths_device_ = nullptr;
    float* inv_masses_device_ = nullptr;
    float* max_error_device_ = nullptr;

    // State
    int32_t iterations_ = 0;
    float max_error_host_ = 0.0f;
    bool converged_ = false;
};

#endif // SHAKE_HOST_H
