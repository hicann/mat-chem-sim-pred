/**
 * ewald_host.cpp
 *
 * Ewald Reciprocal Sum — Host-Side Implementation
 *
 * Uses RegisterAscendBinary + LaunchAscendKernel.
 * The kernel binary is embedded from libewald_reciprocal_kernel.so's
 * .ascend.kernel section, or directly from device_aiv.o.
 */

#include "ewald_host.h"

// ============================================================
// CANN Runtime API
// ============================================================
extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim,
                            void **args, uint32_t size, const void *stream);
int UnregisterAscendBinary(void *hdl);
}

// ============================================================
// Embedded kernel binary (placeholder — will be auto-generated)
// ============================================================
static const unsigned char ewald_reciprocal_kernel_bin[] = {
#include "ewald_device_aiv_bin.inc"
};
static const uint64_t ewald_reciprocal_kernel_len = sizeof(ewald_reciprocal_kernel_bin);

// Registered handle
static void* g_ewald_handle = nullptr;
static bool g_ewald_loaded = false;

// ============================================================
// Initialize kernel binary
// ============================================================
static int32_t EwaldHost_InitKernel() {
    if (g_ewald_loaded) return 0;

    uint32_t ret = RegisterAscendBinary(
        (const char*)ewald_reciprocal_kernel_bin,
        ewald_reciprocal_kernel_len,
        1,
        &g_ewald_handle
    );

    if (ret != 0 || !g_ewald_handle) {
        fprintf(stderr, "[Ewald] FATAL: RegisterAscendBinary failed (ret=%u)\n", ret);
        return -1;
    }

    fprintf(stdout, "[Ewald] Kernel registered (handle=%p, size=%llu)\n",
            g_ewald_handle, (unsigned long long)ewald_reciprocal_kernel_len);
    g_ewald_loaded = true;
    return 0;
}

// ============================================================
// Launch kernel
// ============================================================
static int32_t EwaldHost_LaunchKernel(void* args, uint32_t argsSize, aclrtStream stream) {
    if (!g_ewald_handle) return -1;

    void* overflow_dev = nullptr;
    aclrtMalloc(&overflow_dev, 8, ACL_MEM_MALLOC_HUGE_FIRST);
    if (overflow_dev) {
        uint64_t zero = 0;
        aclrtMemcpy(overflow_dev, 8, &zero, 8, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    struct {
        EwaldForceArgs args;
        void* __ascendc_overflow;
    } __args;

    memcpy(&__args.args, args, sizeof(EwaldForceArgs));
    __args.__ascendc_overflow = overflow_dev;

    uint32_t ret = LaunchAscendKernel(g_ewald_handle, 0, 1,
        (void**)&__args, sizeof(__args), stream);

    if (overflow_dev) aclrtFree(overflow_dev);
    return (int32_t)ret;
}

// ============================================================
// EwaldHost implementation
// ============================================================

int32_t EwaldHost::Initialize(
    int32_t n_atoms, int32_t n_types,
    float box_size, float ewald_alpha, float kmax,
    int32_t device_id,
    aclrtContext external_context,
    aclrtStream external_stream)
{
    if (initialized_) return 0;
    device_id_ = device_id;

    if (!g_ewald_loaded) {
        int32_t ret = EwaldHost_InitKernel();
        if (ret != 0) return ret;
    }

    config_.n_atoms     = n_atoms;
    config_.n_types     = n_types;
    config_.box_size    = box_size;
    config_.half_box    = box_size * 0.5f;
    config_.inv_box     = 1.0f / box_size;
    config_.volume      = box_size * box_size * box_size;
    config_.ewald_alpha = ewald_alpha;
    config_.kmax        = kmax;

    int32_t coords_sz     = n_atoms * 3 * sizeof(float);
    int32_t types_sz      = n_atoms * sizeof(int32_t);
    int32_t type_params_sz = n_types * 3 * sizeof(float);
    int32_t pot_sz        = 3 * sizeof(float);

    pot_host_ = (float*)malloc(pot_sz);

    // Use external context/stream if provided, otherwise create our own
    if (external_context && external_stream) {
        context_ = external_context;
        stream_ = external_stream;
        owns_context_ = false;
        owns_stream_ = false;
    } else {
        aclrtSetDevice(device_id_);
        aclrtContext ctx = nullptr;
        aclrtCreateContext(&ctx, device_id_);
        context_ = ctx;
        aclrtCreateStream(&stream_);
        owns_context_ = true;
        owns_stream_ = true;
    }

    // Allocate device memory (not forces — linked externally)
    if (coords_sz > 0)
        aclrtMalloc(&coords_, coords_sz, ACL_MEM_MALLOC_HUGE_FIRST);
    if (types_sz > 0)
        aclrtMalloc(&types_, types_sz, ACL_MEM_MALLOC_HUGE_FIRST);
    if (type_params_sz > 0)
        aclrtMalloc(&type_params_, type_params_sz, ACL_MEM_MALLOC_HUGE_FIRST);
    if (pot_sz > 0)
        aclrtMalloc(&pot_, pot_sz, ACL_MEM_MALLOC_HUGE_FIRST);

    // Setup GM (forces will be linked separately)
    gm_.coords_gm       = coords_;
    gm_.types_gm        = types_;
    gm_.type_params_gm  = type_params_;
    gm_.forces_gm       = nullptr;  // must be set via LinkGMForces
    gm_.pot_gm          = pot_;

    initialized_ = true;
    return 0;
}

int32_t EwaldHost::SetCoordinates(const float* host_coords) {
    if (!initialized_) return -1;
    int32_t sz = config_.n_atoms * 3 * sizeof(float);
    aclError ret = aclrtMemcpy(coords_, sz, host_coords, sz, ACL_MEMCPY_HOST_TO_DEVICE);
    return (ret == ACL_SUCCESS) ? 0 : -1;
}

int32_t EwaldHost::SetAtomTypes(const int32_t* host_types) {
    if (!initialized_) return -1;
    int32_t sz = config_.n_atoms * sizeof(int32_t);
    aclError ret = aclrtMemcpy(types_, sz, host_types, sz, ACL_MEMCPY_HOST_TO_DEVICE);
    return (ret == ACL_SUCCESS) ? 0 : -1;
}

int32_t EwaldHost::SetTypeParams(const float* host_type_params, int32_t n_types) {
    if (!initialized_) return -1;
    int32_t sz = n_types * 3 * sizeof(float);
    aclError ret = aclrtMemcpy(type_params_, sz, host_type_params, sz, ACL_MEMCPY_HOST_TO_DEVICE);
    return (ret == ACL_SUCCESS) ? 0 : -1;
}

float EwaldHost::ComputeForces() {
    if (!initialized_ || !forces_) {
        fprintf(stderr, "[Ewald] Not initialized or forces not linked\n");
        return 0.0f;
    }

    gm_.forces_gm = forces_;

    EwaldForceArgs args;
    args.config = config_;
    args.gm = gm_;
    args.tile_start = 0;
    args.tile_end = config_.n_atoms;

    int32_t ret = EwaldHost_LaunchKernel(&args, sizeof(args), stream_);
    if (ret != 0) {
        fprintf(stderr, "[Ewald] Kernel launch failed\n");
        return 0.0f;
    }

    aclrtSynchronizeStream(stream_);

    int32_t pot_sz = 3 * sizeof(float);
    aclrtMemcpy(pot_host_, pot_sz, pot_, pot_sz, ACL_MEMCPY_DEVICE_TO_HOST);

    return pot_host_[2];  // total electrostatic
}

void EwaldHost::Finalize() {
    if (!initialized_) return;

    if (coords_) aclrtFree(coords_);
    if (types_) aclrtFree(types_);
    if (type_params_) aclrtFree(type_params_);
    if (pot_) aclrtFree(pot_);
    // forces_ is linked, NOT owned

    if (pot_host_) free(pot_host_);

    // Only destroy context/stream if we own them
    if (owns_stream_ && stream_) aclrtDestroyStream(stream_);
    if (owns_context_ && context_) aclrtDestroyContext(context_);

    coords_ = types_ = type_params_ = forces_ = pot_ = nullptr;
    pot_host_ = nullptr;
    stream_ = nullptr;
    context_ = nullptr;
    initialized_ = false;
}
