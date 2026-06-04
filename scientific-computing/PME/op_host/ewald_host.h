/**
 * ewald_host.h
 *
 * Ewald Reciprocal Sum — Host-Side API
 *
 * Provides C++ interface for launching the Ewald reciprocal kernel
 * on Ascend NPU. Designed to be called AFTER the GAFF2 force kernel
 * to add reciprocal-space electrostatic contributions.
 *
 * Usage:
 *   EwaldHost ewald;
 *   ewald.Initialize(n_atoms, n_types, box_size, ewald_alpha, kmax);
 *   ewald.SetCharges(type_params, n_types);  // or use GAFF2 type_params
 *   ewald.SetCoordinates(coords);            // must match GAFF2 coords
 *   ewald.ComputeForces();                   // ADDS to existing GM forces
 *   float e_recip = ewald.GetRecipEnergy();
 *   float e_self  = ewald.GetSelfEnergy();
 *
 * Key behavior: ewald.ComputeForces() ADDS to the forces already in GM,
 * it does NOT overwrite. The caller must have already run the GAFF2 kernel.
 */

#ifndef EWALD_HOST_H
#define EWALD_HOST_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "acl/acl.h"
#include "ewald_types.h"

class EwaldHost {
public:
    EwaldHost()
        : initialized_(false), device_id_(0), context_(nullptr), stream_(nullptr)
        , owns_context_(true), owns_stream_(true)
        , coords_(nullptr), types_(nullptr), type_params_(nullptr)
        , forces_(nullptr), pot_(nullptr)
        , pot_host_(nullptr) {}

    ~EwaldHost() { Finalize(); }

    // ============================================================
    // Initialize
    //
    // If context/stream are provided externally (from GAFF2ForceEvaluator),
    // pass them in. Otherwise EwaldHost creates its own.
    // ============================================================
    int32_t Initialize(
        int32_t n_atoms, int32_t n_types,
        float box_size, float ewald_alpha, float kmax,
        int32_t device_id = 0,
        aclrtContext external_context = nullptr,
        aclrtStream external_stream = nullptr);

    // ============================================================
    // Upload data (called once or when coordinates change)
    // ============================================================
    int32_t SetCoordinates(const float* host_coords);
    int32_t SetAtomTypes(const int32_t* host_types);
    int32_t SetTypeParams(const float* host_type_params, int32_t n_types);

    // ============================================================
    // Link to existing GM buffers from GAFF2 kernel
    //
    // This allows the Ewald kernel to ADD to the forces already
    // computed by the GAFF2 kernel, without redundant H2D copies.
    // ============================================================
    void LinkGMForces(void* forces_gm) { forces_ = forces_gm; }

    // ============================================================
    // MAIN: Compute Ewald reciprocal sum
    // Reads coordinates from device, ADDS forces to existing GM buffer
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
    bool initialized_;
    int32_t device_id_;
    aclrtContext context_;
    aclrtStream stream_;
    bool owns_context_;
    bool owns_stream_;

    EwaldConfig config_;
    EwaldKernelGM gm_;

    // Device memory
    void* coords_;
    void* types_;
    void* type_params_;
    void* forces_;
    void* pot_;

    // Host memory
    float* pot_host_;
};

#endif // EWALD_HOST_H
