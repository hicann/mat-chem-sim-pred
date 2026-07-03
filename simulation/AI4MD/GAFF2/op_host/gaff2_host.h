/**
 * gaff2_host.h
 *
 * GAFF2 Force Field — Host-Side API
 *
 * Provides C++ interface for launching the GAFF2 kernel on Ascend NPU.
 * Wraps AscendCL buffer management, kernel registration, and launch.
 *
 * Usage:
 *   GAFF2Host gaff2;
 *   gaff2.Initialize(n_atoms, n_types, n_bonds, ...);
 *   gaff2.SetCoordinates(coords);
 *   gaff2.ComputeForces();
 *   float total_energy = gaff2.GetPotentialEnergy();
 *   float* forces = gaff2.GetForces();
 *
 * Memory layout (GM buffers):
 *   coords_gm:      float[n_atoms][3]
 *   types_gm:       int32_t[n_atoms]
 *   bonds_gm:       int32_t[n_bonds][6] (stride=6)
 *   angles_gm:      int32_t[n_angles][6]
 *   dihedrals_gm:   int32_t[n_dihedrals][6]
 *   type_params_gm: float[n_types][3] = {sigma, epsilon, charge}
 *   bond_params_gm: float[n_bond_types][4] = {k, r0, _, _}
 *   angle_params_gm:float[n_angle_types][4] = {k, theta0, _, _}
 *   dihedral_params_gm: float[n_dihedral_types][22] = {Vn[6], per[6], phase[6], n_terms, pad[3]}
 *   forces_gm:      float[n_atoms][3]
 *   pot_gm:         float[5] = {total, bond, angle, dihedral, nb}
 *   virial_gm:      float[6] = {xx, yy, zz, xy, xz, yz}
 */

#ifndef GAFF2_HOST_H
#define GAFF2_HOST_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "acl/acl.h"
#include "gaff2_types.h"

// Error checking macro
#define CHECK_ACL(call, msg) do { \
    aclError ret = (call); \
    if (ret != ACL_SUCCESS) { \
        fprintf(stderr, "[GAFF2 Host] ERROR: %s (ret=%d)\n", msg, ret); \
        return -1; \
    } \
} while(0)

class GAFF2Host {
public:
    GAFF2Host()
        : initialized_(false), device_id_(0), context_(nullptr), stream_(nullptr)
        , coords_(nullptr), types_(nullptr), bonds_(nullptr)
        , angles_(nullptr), dihedrals_(nullptr)
        , type_params_(nullptr), bond_params_(nullptr)
        , angle_params_(nullptr), dihedral_params_(nullptr)
        , forces_(nullptr), pot_(nullptr), virial_(nullptr), exclusion_(nullptr)
        , coords_host_(nullptr), forces_host_(nullptr)
        , pot_host_(nullptr), virial_host_(nullptr)
    {
        memset(&config_, 0, sizeof(config_));
        memset(&gm_, 0, sizeof(gm_));
    }

    ~GAFF2Host() {
        Finalize();
    }

    // ============================================================
    // Initialize the GAFF2 force field evaluator
    // Called once at simulation start
    //
    // ewald_alpha: PME Ewald short-range parameter (nm⁻¹)
    //   If > 0, enables erfc(αr)/r Coulomb (LJ_PME mode)
    //   If <= 0, uses plain 1/r Coulomb (original mode)
    // ============================================================
    int32_t Initialize(
        int32_t n_atoms, int32_t n_types,
        int32_t n_bonds, int32_t n_angles, int32_t n_dihedrals,
        float box_size, float cutoff,
        float lj_14_scale = 0.5f, float coul_14_scale = 0.83333f,
        float ewald_alpha = 0.0f,
        int32_t device_id = 0);

    // ============================================================
    // Update the number of bond/angle/dihedral types based on actual data
    // ============================================================
    void SetNBondTypes(int32_t n) { config_.n_bond_types = n; }
    void SetNAngleTypes(int32_t n) { config_.n_angle_types = n; }
    void SetNDihedralTypes(int32_t n) { config_.n_dihedral_types = n; }

    // ============================================================
    // Upload topology data (bonds, angles, dihedrals) to device
    // ============================================================
    int32_t UploadBonds(const int32_t* host_bonds, int32_t n_bonds) {
        if (!initialized_) return -1;
        int32_t sz = n_bonds * 6 * sizeof(int32_t);
        CHECK_ACL(aclrtMemcpy(bonds_, sz, host_bonds, sz, ACL_MEMCPY_HOST_TO_DEVICE), "UploadBonds");
        return 0;
    }

    int32_t UploadAngles(const int32_t* host_angles, int32_t n_angles) {
        if (!initialized_) return -1;
        int32_t sz = n_angles * 6 * sizeof(int32_t);
        CHECK_ACL(aclrtMemcpy(angles_, sz, host_angles, sz, ACL_MEMCPY_HOST_TO_DEVICE), "UploadAngles");
        return 0;
    }

    int32_t UploadDihedrals(const int32_t* host_dihedrals, int32_t n_dihedrals) {
        if (!initialized_) return -1;
        int32_t sz = n_dihedrals * 6 * sizeof(int32_t);
        CHECK_ACL(aclrtMemcpy(dihedrals_, sz, host_dihedrals, sz, ACL_MEMCPY_HOST_TO_DEVICE), "UploadDihedrals");
        return 0;
    }

    // ============================================================
    // Upload force field parameters to device
    // ============================================================
    int32_t UploadTypeParams(const float* host_params, int32_t n_types) {
        if (!initialized_) return -1;
        int32_t sz = n_types * 3 * sizeof(float);
        CHECK_ACL(aclrtMemcpy(type_params_, sz, host_params, sz, ACL_MEMCPY_HOST_TO_DEVICE), "UploadTypeParams");
        return 0;
    }

    int32_t UploadBondParams(const float* host_params, int32_t n_types) {
        if (!initialized_) return -1;
        int32_t sz = n_types * 4 * sizeof(float);
        CHECK_ACL(aclrtMemcpy(bond_params_, sz, host_params, sz, ACL_MEMCPY_HOST_TO_DEVICE), "UploadBondParams");
        config_.n_bond_types = n_types;
        return 0;
    }

    int32_t UploadAngleParams(const float* host_params, int32_t n_types) {
        if (!initialized_) return -1;
        int32_t sz = n_types * 4 * sizeof(float);
        CHECK_ACL(aclrtMemcpy(angle_params_, sz, host_params, sz, ACL_MEMCPY_HOST_TO_DEVICE), "UploadAngleParams");
        config_.n_angle_types = n_types;
        return 0;
    }

    int32_t UploadDihedralParams(const float* host_params, int32_t n_types) {
        if (!initialized_) return -1;
        constexpr int32_t DIH_STRIDE = 22;
        int32_t sz = n_types * DIH_STRIDE * sizeof(float);
        CHECK_ACL(aclrtMemcpy(dihedral_params_, sz, host_params, sz, ACL_MEMCPY_HOST_TO_DEVICE), "UploadDihedralParams");
        config_.n_dihedral_types = n_types;
        return 0;
    }

    // ============================================================
    // Upload exclusion mask to device.
    // Must be called after bonds/angles/dihedrals are uploaded.
    // exclusion is NxN int32_t where:
    //   0 = normal pair
    //   1 = 1-4 pair (apply lj_14_scale / coul_14_scale)
    //   2 = 1-2 or 1-3 pair (fully excluded)
    // ============================================================
    int32_t UploadExclusion(const int32_t* host_exclusion, int32_t n_atoms) {
        if (!initialized_ || !exclusion_) return -1;
        // CWE-120 fix: exclusion_ was allocated for config_.n_atoms. The caller-
        // supplied n_atoms must match, otherwise a larger value overflows the
        // buffer (heap overflow) and a smaller value leaves stale data.
        if (n_atoms != config_.n_atoms) {
            fprintf(stderr, "[GAFF2] UploadExclusion: n_atoms(%d) != initialized n_atoms(%d)\n",
                    n_atoms, config_.n_atoms);
            return -1;
        }
        // CWE-190 fix: compute the size in 64-bit arithmetic to avoid overflow.
        int64_t sz = static_cast<int64_t>(config_.n_atoms) * config_.n_atoms * sizeof(int32_t);
        CHECK_ACL(aclrtMemcpy(exclusion_, (size_t)sz, host_exclusion, (size_t)sz,
                              ACL_MEMCPY_HOST_TO_DEVICE), "UploadExclusion");
        return 0;
    }

    // ============================================================
    // Set coordinates (host → device copy)
    // ============================================================
    int32_t SetCoordinates(const float* host_coords) {
        if (!initialized_) return -1;
        int32_t sz = config_.n_atoms * 3 * sizeof(float);
        memcpy(coords_host_, host_coords, sz);
        CHECK_ACL(aclrtMemcpy(coords_, sz, host_coords, sz, ACL_MEMCPY_HOST_TO_DEVICE), "SetCoordinates");
        return 0;
    }

    // ============================================================
    // Set atom types (host → device copy)
    // ============================================================
    int32_t SetAtomTypes(const int32_t* host_types) {
        if (!initialized_) return -1;
        int32_t sz = config_.n_atoms * sizeof(int32_t);
        CHECK_ACL(aclrtMemcpy(types_, sz, host_types, sz, ACL_MEMCPY_HOST_TO_DEVICE), "SetAtomTypes");
        return 0;
    }

    // ============================================================
    // MAIN ENTRY: Compute all GAFF2 forces on NPU
    // Returns total potential energy (kJ/mol)
    // ============================================================
    float ComputeForces();

    // ============================================================
    // Access results
    // ============================================================
    float GetPotentialEnergy()      const { return pot_host_[0]; }
    float GetBondEnergy()           const { return pot_host_[1]; }
    float GetAngleEnergy()          const { return pot_host_[2]; }
    float GetDihedralEnergy()       const { return pot_host_[3]; }
    float GetNonbondedEnergy()      const { return pot_host_[4]; }
    const float* GetForces()        const { return forces_host_; }
    const float* GetVirial()        const { return virial_host_; }
    const float* GetCoordinatesHostPtr() const { return coords_host_; }

    // Get device pointer for use by other kernels (e.g., VV integrator)
    void* GetCoordinatesDevicePtr() const { return coords_; }
    void* GetForcesDevicePtr()      const { return forces_; }
    void* GetPotDevicePtr()         const { return pot_; }

    // Get configuration values for use by other components
    int32_t GetNAtoms()   const { return config_.n_atoms; }
    float   GetBoxSize()  const { return config_.box_size; }
    float   GetHalfBox()  const { return config_.half_box; }
    float   GetInvBox()   const { return config_.inv_box; }
    float   GetCutoff()   const { return config_.cutoff; }

    // Update coordinates on device from host buffer (for SD updates made on host)
    int32_t UploadCoordinates() {
        if (!initialized_) return -1;
        int32_t sz = config_.n_atoms * 3 * sizeof(float);
        CHECK_ACL(aclrtMemcpy(coords_, sz, coords_host_, sz, ACL_MEMCPY_HOST_TO_DEVICE), "UploadCoordinates");
        return 0;
    }

    // Copy forces to user buffer
    void CopyForces(float* dst) const {
        int32_t sz = config_.n_atoms * 3 * sizeof(float);
        memcpy(dst, forces_host_, sz);
    }

    // ============================================================
    // Cleanup
    // ============================================================
    void Finalize();

    // Check if initialized
    bool IsInitialized() const { return initialized_; }

    // Expose context/stream for sharing with other kernels (e.g., EwaldHost)
    // Critical: all kernels on the same device must use the SAME context/stream
    // to avoid AscendCL context conflicts that cause LaunchAscendKernel to fail.
    aclrtContext GetContext() const { return context_; }
    aclrtStream  GetStream()  const { return stream_; }

private:
    bool initialized_;
    int32_t device_id_;
    aclrtContext context_;
    aclrtStream stream_;

    GAFF2Config config_;
    GAFF2KernelGM gm_;

    // Device memory (GM)
    void* coords_;
    void* types_;
    void* bonds_;
    void* angles_;
    void* dihedrals_;
    void* type_params_;
    void* bond_params_;
    void* angle_params_;
    void* dihedral_params_;
    void* forces_;
    void* pot_;
    void* virial_;
    void* exclusion_;

    // Host memory (for results)
    float* coords_host_;
    float* forces_host_;
    float* pot_host_;
    float* virial_host_;
};

#endif // GAFF2_HOST_H
