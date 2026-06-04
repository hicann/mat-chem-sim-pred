/**
 * pme_host.h
 *
 * PME (Particle Mesh Ewald) 3D FFT — Host-Side API
 *
 * Provides C++ interface for launching the PME kernel
 * on Ascend NPU. Designed to be called AFTER the GAFF2 force
 * kernel to add reciprocal-space electrostatic contributions.
 *
 * Usage:
 *   PMEHost pme;
 *   pme.Initialize(n_atoms, box_size, ewald_alpha, mesh_dim);
 *   pme.SetCharges(charges, n_atoms);
 *   pme.SetCoordinates(coords);
 *   pme.PrecomputeDFT();        // Compute DFT matrix + influence function
 *   pme.LinkGMForces(gaff2_forces_gm);
 *   float e_total = pme.ComputeForces();  // spread → FFT → influence → IFFT → interpolate
 *
 * Architecture:
 *   Host prepares: DFT matrix, influence function, charges, coords
 *   Kernel executes: entire PME pipeline on NPU
 *   Host reads: energies from pot_gm
 */

#ifndef PME_HOST_H
#define PME_HOST_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "acl/acl.h"
#include "pme_types.h"

class PMEHost {
public:
    PMEHost()
        : initialized_(false), device_id_(0), context_(nullptr), stream_(nullptr)
        , owns_context_(true), owns_stream_(true)
        , coords_(nullptr), charges_(nullptr), forces_(nullptr)
        , mesh_re_(nullptr), mesh_im_(nullptr)
        , dft_matrix_(nullptr), influence_(nullptr), pot_(nullptr)
        , pot_host_(nullptr)
        , host_coords_(nullptr), host_charges_(nullptr)
        , host_dft_matrix_(nullptr), host_influence_(nullptr)
        , n_atoms_(0), mesh_dim_(0) {}

    ~PMEHost() { Finalize(); }

    // ============================================================
    // Initialize
    //
    // Precomputes DFT matrix and influence function on the host
    // and uploads them to device GM.
    // ============================================================
    int32_t Initialize(
        int32_t n_atoms,
        float box_size,
        float ewald_alpha,
        int32_t mesh_dim = 32,
        int32_t spline_order = 4,
        int32_t device_id = 0,
        aclrtContext external_context = nullptr,
        aclrtStream external_stream = nullptr);

    // ============================================================
    // Upload data
    // ============================================================
    int32_t SetCoordinates(const float* host_coords);
    int32_t SetCharges(const float* host_charges, int32_t n_atoms);

    // ============================================================
    // Link to existing GM buffers from GAFF2 kernel
    // ============================================================
    void LinkGMForces(void* forces_gm) { forces_ = forces_gm; }

    // ============================================================
    // Precompute DFT matrices (call once after Initialize)
    // ============================================================
    int32_t PrecomputeDFT();

    // ============================================================
    // MAIN: Compute PME reciprocal-space forces
    // Reads coordinates from device, executes full PME pipeline,
    // ADDS forces to linked GM buffer.
    // ============================================================
    float ComputeForces();

    // ============================================================
    // Access results
    // ============================================================
    float GetRecipEnergy() const { return pot_host_ ? pot_host_[0] : 0.0f; }
    float GetSelfEnergy()  const { return pot_host_ ? pot_host_[1] : 0.0f; }
    float GetTotalElectrostaticEnergy() const { return pot_host_ ? pot_host_[2] : 0.0f; }

    // ============================================================
    // Cleanup
    // ============================================================
    void Finalize();

    bool IsInitialized() const { return initialized_; }

private:
    // Initialize kernel binary
    static int32_t InitKernel();
    static void*   kernel_handle_;
    static bool    kernel_loaded_;

    // Launch kernel
    int32_t LaunchKernel(void* args, uint32_t argsSize);

    // Precompute DFT matrix (M×M complex)
    void compute_dft_matrix(int32_t M, float* dft_re, float* dft_im);

    // Precompute PME influence function B(m)/m²
    void compute_influence(int32_t M, float box_size, float ewald_alpha,
                          float* influence);

    bool initialized_;
    int32_t device_id_;
    aclrtContext context_;
    aclrtStream stream_;
    bool owns_context_;
    bool owns_stream_;

    PMEConfig config_;
    PMEGlobalMemory gm_;

    // Device memory
    void* coords_;
    void* charges_;
    void* forces_;
    void* mesh_re_;
    void* mesh_im_;
    void* dft_matrix_;
    void* influence_;
    void* pot_;

    // Host memory
    float* pot_host_;
    float* host_coords_;
    float* host_charges_;
    float* host_dft_matrix_;
    float* host_influence_;

    int32_t n_atoms_;
    int32_t mesh_dim_;
};

#endif // PME_HOST_H
