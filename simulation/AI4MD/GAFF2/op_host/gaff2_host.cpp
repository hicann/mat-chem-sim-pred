/**
 * gaff2_host.cpp
 *
 * GAFF2 Force Field — Host-Side Implementation
 *
 * Uses AscendCL standard API (aclrtBinaryLoad + aclrtLaunchKernel) to
 * load and launch the GAFF2 kernel binary on Ascend NPU.
 * The kernel binary is embedded directly from the compiled .so's .ascend.kernel section.
 */

#include "gaff2_host.h"

// ============================================================
// CANN Runtime API declarations (resolved from libascendc_runtime.a)
// ============================================================
extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim,
                            void **args, uint32_t size, const void *stream);
int UnregisterAscendBinary(void *hdl);
}

// ============================================================
// GAFF2 Force Field Kernel Binary
// Binary format: raw ELF object file (device_aiv.o, no extra header)
// Source: gaff2_force_kernel_aiv_device_dir/device_aiv.o (8624 bytes)
// ============================================================
static const unsigned char gaff2_force_kernel_bin[] = {
#include "gaff2_device_aiv_bin.inc"
};
static const uint64_t gaff2_force_kernel_len = sizeof(gaff2_force_kernel_bin);

// The kernel name as embedded in the binary (from section name)
static const char* GAFF2_KERNEL_NAME = "gaff2_compute_forces_0";

// Registered handles (one-time global)
static void* g_kernel_handle = nullptr;
static bool g_kernel_loaded = false;

// ============================================================
// Initialize kernel binary registration — RegisterAscendBinary
// ============================================================
int32_t GAFF2Host_InitKernel() {
    if (g_kernel_loaded) return 0;

    // Register the kernel binary with type=1 (AIV kernel)
    uint32_t ret = RegisterAscendBinary(
        (const char*)gaff2_force_kernel_bin,
        gaff2_force_kernel_len,
        1,
        &g_kernel_handle
    );

    if (ret != 0 || !g_kernel_handle) {
        fprintf(stderr, "[GAFF2] FATAL: RegisterAscendBinary failed (ret=%u, handle=%p)\n",
                ret, g_kernel_handle);
        return -1;
    }

    fprintf(stdout, "[GAFF2] Kernel registered via RegisterAscendBinary (handle=%p, size=%llu)\n",
            g_kernel_handle, (unsigned long long)gaff2_force_kernel_len);
    g_kernel_loaded = true;
    return 0;
}

// ============================================================
// Kernel launch — called by GAFF2Host::ComputeForces
// ============================================================
// ============================================================
// Static overflow buffer — allocated ONCE, reused across all
// kernel launches.  Previous per-launch alloc/free cycle caused
// device memory fragmentation → double-free crash after ~200 calls.
// ============================================================
static void* g_overflow_dev = nullptr;

int32_t GAFF2Host_LaunchKernel(void* args, uint32_t argsSize, aclrtStream stream) {
    if (!g_kernel_handle) {
        fprintf(stderr, "[GAFF2] Kernel handle is null\n");
        return -1;
    }

    // CRITICAL: The auto-gen host_stub structure is:
    //   struct { GAFF2ForceArgs args; void* __ascendc_overflow; }
    // which is sizeof(GAFF2ForceArgs) + 8 = 200 bytes.
    // The kernel entry takes GAFF2ForceArgs BY VALUE on the stack.
    // The runtime copies the full struct from host to kernel stack.
    // This is DIFFERENT from PUFF which passes {void* arguments, overflow}
    // and reads args from a device pointer. Here GAFF2ForceArgs is embedded
    // directly in the launch args block.

    // One-time allocation of overflow buffer (64 bytes, zero-init)
    if (!g_overflow_dev) {
        constexpr uint32_t overflow_size = 64;
        aclrtMalloc(&g_overflow_dev, overflow_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (g_overflow_dev) {
            uint64_t zero = 0;
            aclrtMemcpy(g_overflow_dev, overflow_size, &zero, overflow_size, ACL_MEMCPY_HOST_TO_DEVICE);
        }
    }

    // We need to build: {GAFF2ForceArgs, void*} with correct alignment
    struct {
        GAFF2ForceArgs args;
        void* __ascendc_overflow;
    } __ascendc_args;

    // Copy args from caller
    memcpy(&__ascendc_args.args, args, sizeof(GAFF2ForceArgs));
    __ascendc_args.__ascendc_overflow = g_overflow_dev;

    fprintf(stdout, "[GAFF2] Launch (wrapper_sz=%zu)...\n", sizeof(__ascendc_args));
    uint32_t ret = LaunchAscendKernel(g_kernel_handle, 0, 1,
        (void**)&__ascendc_args, sizeof(__ascendc_args), stream);
    fprintf(stdout, "[GAFF2] Launch ret=%u\n", ret);

    return (int32_t)ret;
}

// ============================================================
// GAFF2Host class implementation
// ============================================================

int32_t GAFF2Host::Initialize(
    int32_t n_atoms, int32_t n_types,
    int32_t n_bonds, int32_t n_angles, int32_t n_dihedrals,
    float box_size, float cutoff,
    float lj_14_scale, float coul_14_scale,
    float ewald_alpha,
    int32_t device_id)
{
    if (initialized_) return 0;

    device_id_ = device_id;

    // One-time kernel loading
    if (!g_kernel_loaded) {
        int32_t ret = GAFF2Host_InitKernel();
        if (ret != 0) return ret;
    }

    // Set config
    config_.ff_class = GAFF2;
    config_.nb_type = (ewald_alpha > 0.0f) ? LJ_PME : LJ_CUT;
    config_.bond_type = HARMONIC;
    config_.angle_type = HARMONIC_ANGLE;
    config_.dihedral_type = GAFF2_DIHEDRAL;
    config_.comb_rule = LORENTZ_BERTHELOT;
    config_.exclusion = BIJKL;

    config_.n_atoms     = n_atoms;
    config_.n_types     = n_types;
    config_.n_bonds     = n_bonds;
    config_.n_angles    = n_angles;
    config_.n_dihedrals = n_dihedrals;
    config_.n_impropers = 0;
    config_.n_bond_types = n_bonds > 0 ? 1 : 0;
    config_.n_angle_types = n_angles > 0 ? 1 : 0;
    config_.n_dihedral_types = n_dihedrals > 0 ? 1 : 0;
    config_.n_improper_types = 0;

    config_.box_size   = box_size;
    config_.half_box   = box_size * 0.5f;
    config_.inv_box    = 1.0f / box_size;
    config_.cutoff     = cutoff;
    config_.cutoff_sq  = cutoff * cutoff;
    config_.lj_14_scale   = lj_14_scale;
    config_.coul_14_scale = coul_14_scale;
    config_.ewald_alpha   = ewald_alpha;

    // Calculate buffer sizes
    int32_t coords_sz    = n_atoms * 3 * sizeof(float);
    int32_t types_sz     = n_atoms * sizeof(int32_t);
    int32_t bonds_sz     = n_bonds * 6 * sizeof(int32_t);
    int32_t angles_sz    = n_angles * 6 * sizeof(int32_t);
    int32_t dihedrals_sz = n_dihedrals * 6 * sizeof(int32_t);
    int32_t type_params_sz  = n_types * 3 * sizeof(float);
    int32_t bond_params_sz  = (n_bonds > 0 ? 1 : 0) * 4 * sizeof(float);
    int32_t angle_params_sz = (n_angles > 0 ? 1 : 0) * 4 * sizeof(float);
    constexpr int32_t DIH_PARAM_STRIDE = 22;
    int32_t dihedral_params_sz = (n_dihedrals > 0 ? 1 : 0) * DIH_PARAM_STRIDE * sizeof(float);
    int32_t forces_sz     = n_atoms * 3 * sizeof(float);
    int32_t pot_sz        = 5 * sizeof(float);
    int32_t virial_sz     = 6 * sizeof(float);

    // Allocate host memory
    coords_host_    = (float*)malloc(coords_sz);
    forces_host_    = (float*)malloc(forces_sz);
    pot_host_       = (float*)malloc(pot_sz);
    virial_host_    = (float*)malloc(virial_sz);

    if (!coords_host_ || !forces_host_ || !pot_host_ || !virial_host_) {
        fprintf(stderr, "GAFF2 Host: Failed to allocate host memory\n");
        return -1;
    }

    // Set device and create stream
    CHECK_ACL(aclrtSetDevice(device_id_), "aclrtSetDevice");

    // Create context (recommended for AscendCL operations)
    aclrtContext ctx = nullptr;
    aclError ctx_ret = aclrtCreateContext(&ctx, device_id_);
    if (ctx_ret != ACL_SUCCESS) {
        fprintf(stderr, "GAFF2 Host: Failed to create context: %d\n", ctx_ret);
        return -1;
    }
    context_ = ctx;

    CHECK_ACL(aclrtCreateStream(&stream_), "aclrtCreateStream");

    // Allocate device memory
    if (coords_sz > 0)
        CHECK_ACL(aclrtMalloc(&coords_,         coords_sz,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc coords");
    if (types_sz > 0)
        CHECK_ACL(aclrtMalloc(&types_,          types_sz,     ACL_MEM_MALLOC_HUGE_FIRST), "Malloc types");
    if (bonds_sz > 0)
        CHECK_ACL(aclrtMalloc(&bonds_,          bonds_sz,     ACL_MEM_MALLOC_HUGE_FIRST), "Malloc bonds");
    if (angles_sz > 0)
        CHECK_ACL(aclrtMalloc(&angles_,         angles_sz,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc angles");
    if (dihedrals_sz > 0)
        CHECK_ACL(aclrtMalloc(&dihedrals_,      dihedrals_sz, ACL_MEM_MALLOC_HUGE_FIRST), "Malloc dihedrals");
    if (type_params_sz > 0)
        CHECK_ACL(aclrtMalloc(&type_params_,    type_params_sz,   ACL_MEM_MALLOC_HUGE_FIRST), "Malloc type_params");
    if (bond_params_sz > 0)
        CHECK_ACL(aclrtMalloc(&bond_params_,    bond_params_sz,   ACL_MEM_MALLOC_HUGE_FIRST), "Malloc bond_params");
    if (angle_params_sz > 0)
        CHECK_ACL(aclrtMalloc(&angle_params_,   angle_params_sz,  ACL_MEM_MALLOC_HUGE_FIRST), "Malloc angle_params");
    if (dihedral_params_sz > 0)
        CHECK_ACL(aclrtMalloc(&dihedral_params_, dihedral_params_sz, ACL_MEM_MALLOC_HUGE_FIRST), "Malloc dihedral_params");
    if (forces_sz > 0)
        CHECK_ACL(aclrtMalloc(&forces_,         forces_sz,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc forces");
    if (pot_sz > 0)
        CHECK_ACL(aclrtMalloc(&pot_,            pot_sz,       ACL_MEM_MALLOC_HUGE_FIRST), "Malloc pot");
    if (virial_sz > 0)
        CHECK_ACL(aclrtMalloc(&virial_,         virial_sz,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc virial");

    // Allocate exclusion mask buffer (N×N int32_t)
    if (n_atoms > 0) {
        // CWE-190 fix: n_atoms*n_atoms*sizeof(int32_t) overflows int32_t when
        // n_atoms > ~23170. Compute the size in 64-bit arithmetic and validate.
        int64_t excl_sz = static_cast<int64_t>(n_atoms) * n_atoms * sizeof(int32_t);
        if (excl_sz <= 0) {
            fprintf(stderr, "[GAFF2] Invalid exclusion buffer size (n_atoms=%d)\n", n_atoms);
            Finalize();
            return -1;
        }
        CHECK_ACL(aclrtMalloc(&exclusion_, (size_t)excl_sz, ACL_MEM_MALLOC_HUGE_FIRST), "Malloc exclusion");
        // Initialize to zeros (all pairs = normal, no exclusion)
        // This ensures that if UploadExclusion is not called, the kernel sees valid data
        aclrtMemset(exclusion_, (size_t)excl_sz, 0, (size_t)excl_sz);
        fprintf(stdout, "[GAFF2] exclusion_ buffer allocated: %lld bytes at %p\n",
                (long long)excl_sz, exclusion_);
    }

    // Setup GM structure
    gm_.coords_gm          = coords_;
    gm_.types_gm           = types_;
    gm_.bonds_gm           = bonds_;
    gm_.angles_gm          = angles_;
    gm_.dihedrals_gm       = dihedrals_;
    gm_.type_params_gm     = type_params_;
    gm_.bond_params_gm     = bond_params_;
    gm_.angle_params_gm    = angle_params_;
    gm_.dihedral_params_gm = dihedral_params_;
    gm_.exclusion_gm       = exclusion_;  // <-- NEW: assign exclusion pointer
    gm_.forces_gm          = forces_;
    gm_.pot_gm             = pot_;
    gm_.virial_gm          = virial_;

    initialized_ = true;
    return 0;
}

float GAFF2Host::ComputeForces() {
    if (!initialized_) return 0.0f;

    // Prepare argument struct for kernel
    GAFF2ForceArgs args;
    args.config = config_;
    args.gm = gm_;
    args.tile_start = 0;
    args.tile_end = config_.n_atoms;

    // Launch kernel
    int32_t ret = GAFF2Host_LaunchKernel(&args, sizeof(args), stream_);
    if (ret != 0) {
        fprintf(stderr, "[GAFF2] ComputeForces: kernel launch failed\n");
        return 0.0f;
    }

    // Synchronize
    aclrtSynchronizeStream(stream_);

    // Download results
    int32_t pot_sz = 5 * sizeof(float);
    aclrtMemcpy(pot_host_, pot_sz, pot_, pot_sz, ACL_MEMCPY_DEVICE_TO_HOST);

    int32_t forces_sz = config_.n_atoms * 3 * sizeof(float);
    aclrtMemcpy(forces_host_, forces_sz, forces_, forces_sz, ACL_MEMCPY_DEVICE_TO_HOST);

    int32_t virial_sz = 6 * sizeof(float);
    aclrtMemcpy(virial_host_, virial_sz, virial_, virial_sz, ACL_MEMCPY_DEVICE_TO_HOST);

    return pot_host_[0];
}

void GAFF2Host::Finalize() {
    if (!initialized_) return;

    if (coords_)  aclrtFree(coords_);
    if (types_)   aclrtFree(types_);
    if (bonds_)   aclrtFree(bonds_);
    if (angles_)  aclrtFree(angles_);
    if (dihedrals_) aclrtFree(dihedrals_);
    if (type_params_)    aclrtFree(type_params_);
    if (bond_params_)    aclrtFree(bond_params_);
    if (angle_params_)   aclrtFree(angle_params_);
    if (dihedral_params_) aclrtFree(dihedral_params_);
    if (forces_)  aclrtFree(forces_);
    if (pot_)     aclrtFree(pot_);
    if (virial_)  aclrtFree(virial_);
    if (exclusion_) aclrtFree(exclusion_);

    if (coords_host_)  free(coords_host_);
    if (forces_host_)  free(forces_host_);
    if (pot_host_)     free(pot_host_);
    if (virial_host_)  free(virial_host_);

    if (stream_) aclrtDestroyStream(stream_);
    if (context_) aclrtDestroyContext(context_);
    aclrtResetDevice(device_id_);

    coords_ = bonds_ = angles_ = dihedrals_ = forces_ = pot_ = virial_ = nullptr;
    types_ = type_params_ = bond_params_ = angle_params_ = dihedral_params_ = nullptr;
    exclusion_ = nullptr;
    coords_host_ = forces_host_ = pot_host_ = virial_host_ = nullptr;
    stream_ = nullptr;
    context_ = nullptr;
    initialized_ = false;
}
