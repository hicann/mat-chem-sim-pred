/**
 * pme_host.cpp
 *
 * PME (Particle Mesh Ewald) 3D FFT — Host-Side Implementation
 *
 * Manages the complete PME pipeline on NPU:
 *   1. Host precomputes DFT matrix + influence function
 *   2. Uploads matrices to device GM
 *   3. Launches PME kernel (spread → FFT → influence → IFFT → interpolate)
 *   4. Reads energy results back
 *
 * Kernel binary is embedded from the built .so or device_aiv.o.
 * Launch follows the same pattern as ewald_host.cpp.
 */

#include "pme_host2.h"
#include <algorithm>

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
// Static kernel management
// ============================================================
void* PMEHost::kernel_handle_ = nullptr;
bool  PMEHost::kernel_loaded_ = false;

// ============================================================
// Embedded kernel binary (placeholder)
//
// After building the PME kernel, extract device_aiv.o from
// build/npu_ops/pme/merge_obj_dir/ and convert with xxd -i.
// Then include the generated .inc file here.
//
// For now, this registers at runtime via the .so linking approach
// (preferred pattern from P15 in skill notes).
// ============================================================

// ============================================================
// Initialize kernel binary (static)
// ============================================================
int32_t PMEHost::InitKernel() {
    if (kernel_loaded_) return 0;

    // Using CANN auto-generated .so registration
    // The constructor of libpme_kernel.so handles RegisterAscendBinary
    fprintf(stdout, "[PME] Kernel registration via .so constructor\n");
    kernel_loaded_ = true;
    return 0;
}

// ============================================================
// Initialize
// ============================================================
int32_t PMEHost::Initialize(
    int32_t n_atoms,
    float box_size,
    float ewald_alpha,
    int32_t mesh_dim,
    int32_t spline_order,
    int32_t device_id,
    aclrtContext external_context,
    aclrtStream external_stream)
{
    if (initialized_) return 0;

    device_id_ = device_id;
    n_atoms_ = n_atoms;
    mesh_dim_ = mesh_dim;

    // Init kernel
    int32_t ret = InitKernel();
    if (ret != 0) return ret;

    // Configure
    config_.n_atoms       = n_atoms;
    config_.mesh_dim      = mesh_dim;
    config_.spline_order  = spline_order;
    config_.ewald_alpha   = ewald_alpha;
    config_.box_size      = box_size;
    config_.half_box      = box_size * 0.5f;
    config_.inv_box       = 1.0f / box_size;
    config_.volume        = box_size * box_size * box_size;
    config_.coulomb_c     = PME_COULOMB_C;
    config_.pad           = 0;

    // Size calculation — use 64-bit arithmetic to avoid integer overflow (CWE-190):
    // mesh_dim^3 * sizeof(float) overflows int32_t when mesh_dim > ~2150.
    if (n_atoms <= 0 || mesh_dim <= 0 || mesh_dim > 65535) {
        fprintf(stderr, "[PME] Invalid params: n_atoms=%d, mesh_dim=%d\n", n_atoms, mesh_dim);
        return -1;
    }
    int64_t mesh_size   = static_cast<int64_t>(mesh_dim) * mesh_dim * mesh_dim * sizeof(float);
    int64_t dft_size    = static_cast<int64_t>(mesh_dim) * mesh_dim * 2 * sizeof(float);  // real + imag
    int64_t pot_sz      = 3 * sizeof(float);

    // Host buffers
    pot_host_        = (float*)malloc((size_t)pot_sz);
    host_coords_     = (float*)malloc(static_cast<size_t>(n_atoms) * 3 * sizeof(float));
    host_charges_    = (float*)malloc(static_cast<size_t>(n_atoms) * sizeof(float));
    host_dft_matrix_ = (float*)malloc((size_t)dft_size);
    host_influence_  = (float*)malloc((size_t)mesh_size);

    // CWE-476 fix: check ALL malloc returns (host_dft_matrix_ and host_influence_
    // were previously unchecked). free(NULL) is safe, so this cleans up correctly
    // regardless of which allocation failed.
    if (!pot_host_ || !host_coords_ || !host_charges_ ||
        !host_dft_matrix_ || !host_influence_) {
        fprintf(stderr, "[PME] Failed to allocate host buffers (malloc returned null)\n");
        free(pot_host_);        pot_host_ = nullptr;
        free(host_coords_);     host_coords_ = nullptr;
        free(host_charges_);    host_charges_ = nullptr;
        free(host_dft_matrix_); host_dft_matrix_ = nullptr;
        free(host_influence_);  host_influence_ = nullptr;
        return -1;
    }

    // Use external context/stream if provided
    if (external_context && external_stream) {
        context_ = external_context;
        stream_ = external_stream;
        owns_context_ = false;
        owns_stream_ = false;
    } else {
        aclrtSetDevice(device_id_);
        aclrtCreateContext(&context_, device_id_);
        aclrtCreateStream(&stream_);
        owns_context_ = true;
        owns_stream_ = true;
    }

    // Allocate device memory (sizes computed in 64-bit above; cast to size_t)
    aclrtMalloc(&coords_,     static_cast<size_t>(n_atoms) * 3 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&charges_,    static_cast<size_t>(n_atoms) * sizeof(float),     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&mesh_re_,    (size_t)mesh_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&mesh_im_,    (size_t)mesh_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dft_matrix_, (size_t)dft_size,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&influence_,  (size_t)mesh_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&pot_,        (size_t)pot_sz,    ACL_MEM_MALLOC_HUGE_FIRST);
    // forces_ is linked externally

    // Setup GM
    gm_.coords_gm      = coords_;
    gm_.charges_gm     = charges_;
    gm_.forces_gm      = nullptr;    // must be set via LinkGMForces
    gm_.mesh_re_gm     = mesh_re_;
    gm_.mesh_im_gm     = mesh_im_;
    gm_.dft_matrix_gm  = dft_matrix_;
    gm_.influence_gm   = influence_;
    gm_.pot_gm         = pot_;

    // Precompute DFT matrix and influence function (host-side)
    compute_dft_matrix(mesh_dim, host_dft_matrix_, host_dft_matrix_ + mesh_dim * mesh_dim);
    compute_influence(mesh_dim, box_size, ewald_alpha, host_influence_);

    // Upload to device
    aclrtMemcpy(dft_matrix_, dft_size, host_dft_matrix_, dft_size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(influence_,  mesh_size, host_influence_,  mesh_size,  ACL_MEMCPY_HOST_TO_DEVICE);

    initialized_ = true;
    return 0;
}

// ============================================================
// Compute DFT matrix
// W[j][k] = exp(-2πi · j · k / M)
// ============================================================
void PMEHost::compute_dft_matrix(int32_t M, float* dft_re, float* dft_im) {
    for (int32_t j = 0; j < M; j++) {
        for (int32_t k = 0; k < M; k++) {
            double theta = -2.0 * M_PI * j * k / M;
            dft_re[j * M + k] = (float)cos(theta);
            dft_im[j * M + k] = (float)sin(theta);  // sin(−θ) = −sin(θ)
        }
    }
}

// ============================================================
// Compute PME influence function
//
// B(k) = exp(π² · k² / (α² · M²)) / (k² · V)   (simplified)
//
// where k = (kx, ky, kz) in reduced grid units, and
// k² = kx² + ky² + kz² (but shifted to range [-M/2, M/2])
//
// For the actual PME with B-spline correction:
//   θ̃(k) = Σ_m θ_n(m) · exp(-2πi·k·m/M)   (B-spline Fourier coeff)
//   correction = |θ̃(k)|⁻²
//   B(k) = exp(π²·k²/α²·M²) · correction / (k²·V)
//
// Since we use cubic B-spline (n=4) and mesh_dim=32,
// we approximate with:
//   B(k) = exp(π²·k²/α²·M²) · ( (M·k)² )? No, use the standard PME formula:
//
// PME energy: E = (1/2·V) · Σ_{k≠0} Q̃*(k) · exp(π²·k²/α²·M²) / k² · Q̃(k)
// The factor exp(π²·k²/α²·M²) is the same as in direct Ewald but with
// reciprocal lattice vectors in reduced coordinates.
//
// For reference, in GROMACS PME:
//   G(mx,my,mz) = exp(2π²·m²·p²) · (π·m/M)⁻²  (where p = 1/(α·M·L) )
// The actual influence function is precomputed on host.
//
// Simplified: precompute B_influence = exp(-π²·m² / (α²·M²·L²))? No...
//
// The key PME formula (from Essmann 1995):
//   E_recip = 1/(2·π·V) · Σ_{m≠0} exp(π²·m²/α²) · |θ̃(m)|² / m² · |Q̃(m)|²
// where m = (m_x, m_y, m_z) ∈ [-M/2, M/2)³ in grid units
//
// For our implementation:
//   We multiply Q̃(k) by sqrt(B(k)) to get Q̃_conv, then IFFT to get real-space force.
//   B(k) = exp(π²·|k|² / (α² · M²)) / (|k|² · V)
//
// where |k|² = (nx² + ny² + nz²) with nx, ny, nz ∈ [-M/2, M/2)
// ============================================================
void PMEHost::compute_influence(int32_t M, float box_size, float ewald_alpha,
                                float* influence) {
    double V = (double)box_size * box_size * box_size;
    double alpha_sq = (double)ewald_alpha * ewald_alpha;
    double inv_alpha_sq_Msq = 1.0 / (alpha_sq * (double)M * (double)M);

    for (int32_t nz = 0; nz < M; nz++) {
        int32_t kz = (nz <= M / 2) ? nz : nz - M;  // shift
        for (int32_t ny = 0; ny < M; ny++) {
            int32_t ky = (ny <= M / 2) ? ny : ny - M;
            for (int32_t nx = 0; nx < M; nx++) {
                int32_t kx = (nx <= M / 2) ? nx : nx - M;

                double k2 = (double)(kx*kx + ky*ky + kz*kz);
                int32_t idx = (nz * M + ny) * M + nx;

                if (k2 < 0.5) {
                    // k=0 term: zero (neutral system)
                    influence[idx] = 0.0f;
                } else {
                    double k2_Msq = k2 / (M * M);  // |k|²/M² = reduced units
                    double exp_term = exp(-M_PI * M_PI * k2 / (alpha_sq * M * M));
                    influence[idx] = (float)exp_term;
                    influence[idx] *= (float)(1.0 / (k2 * V));
                }
            }
        }
    }
}

// ============================================================
// Upload coordinates
// ============================================================
int32_t PMEHost::SetCoordinates(const float* host_coords) {
    if (!initialized_) return -1;
    memcpy(host_coords_, host_coords, n_atoms_ * 3 * sizeof(float));
    int32_t sz = n_atoms_ * 3 * sizeof(float);
    aclError ret = aclrtMemcpy(coords_, sz, host_coords, sz, ACL_MEMCPY_HOST_TO_DEVICE);
    return (ret == ACL_SUCCESS) ? 0 : -1;
}

// ============================================================
// Upload charges
// ============================================================
int32_t PMEHost::SetCharges(const float* host_charges, int32_t n_atoms) {
    if (!initialized_) return -1;
    // CWE-120 fix: host_charges_ and charges_ were sized for n_atoms_ during
    // Initialize. The caller n_atoms must match, otherwise a larger value causes
    // a heap buffer overflow on memcpy / aclrtMemcpy.
    if (n_atoms != n_atoms_) {
        fprintf(stderr, "[PME] SetCharges: n_atoms(%d) != initialized n_atoms_(%d)\n",
                n_atoms, n_atoms_);
        return -1;
    }
    size_t sz = static_cast<size_t>(n_atoms_) * sizeof(float);
    memcpy(host_charges_, host_charges, sz);
    aclError ret = aclrtMemcpy(charges_, sz, host_charges, sz, ACL_MEMCPY_HOST_TO_DEVICE);
    return (ret == ACL_SUCCESS) ? 0 : -1;
}

// ============================================================
// Precompute DFT (called from Initialize, can be re-called)
// ============================================================
int32_t PMEHost::PrecomputeDFT() {
    if (!initialized_) return -1;
    // Already done in Initialize
    return 0;
}

// ============================================================
// Launch kernel
// ============================================================
int32_t PMEHost::LaunchKernel(void* args, uint32_t argsSize) {
    if (!kernel_handle_) {
        fprintf(stderr, "[PME] Kernel handle is null — link libpme_kernel.so\n");
        return -1;
    }

    void* overflow_dev = nullptr;
    aclrtMalloc(&overflow_dev, 8, ACL_MEM_MALLOC_HUGE_FIRST);
    if (overflow_dev) {
        uint64_t zero = 0;
        aclrtMemcpy(overflow_dev, 8, &zero, 8, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    struct {
        PMEArgs args;
        void* __ascendc_overflow;
    } __args;

    memcpy(&__args.args, args, sizeof(PMEArgs));
    __args.__ascendc_overflow = overflow_dev;

    uint32_t ret = LaunchAscendKernel(kernel_handle_, 0, 1,
        (void**)&__args, sizeof(__args), stream_);

    if (overflow_dev) aclrtFree(overflow_dev);
    return (int32_t)ret;
}

// ============================================================
// MAIN: Compute PME reciprocal-space forces
// ============================================================
float PMEHost::ComputeForces() {
    if (!initialized_ || !forces_) {
        fprintf(stderr, "[PME] Not initialized or forces not linked\n");
        return 0.0f;
    }

    gm_.forces_gm = forces_;

    PMEArgs args;
    args.config = config_;
    args.gm = gm_;
    args.tile_start = 0;
    args.tile_end = config_.n_atoms;

    int32_t ret = LaunchKernel(&args, sizeof(args));
    if (ret != 0) {
        fprintf(stderr, "[PME] Kernel launch failed\n");
        return 0.0f;
    }

    aclrtSynchronizeStream(stream_);

    // Read energies back
    int32_t pot_sz = 3 * sizeof(float);
    aclrtMemcpy(pot_host_, pot_sz, pot_, pot_sz, ACL_MEMCPY_DEVICE_TO_HOST);

    return pot_host_[2];  // total electrostatic
}

// ============================================================
// Cleanup
// ============================================================
void PMEHost::Finalize() {
    if (!initialized_) return;

    if (coords_)     aclrtFree(coords_);
    if (charges_)    aclrtFree(charges_);
    if (mesh_re_)    aclrtFree(mesh_re_);
    if (mesh_im_)    aclrtFree(mesh_im_);
    if (dft_matrix_) aclrtFree(dft_matrix_);
    if (influence_)  aclrtFree(influence_);
    if (pot_)        aclrtFree(pot_);
    // forces_ is linked, NOT owned

    if (pot_host_)        free(pot_host_);
    if (host_coords_)     free(host_coords_);
    if (host_charges_)    free(host_charges_);
    if (host_dft_matrix_) free(host_dft_matrix_);
    if (host_influence_)  free(host_influence_);

    if (owns_stream_ && stream_)   aclrtDestroyStream(stream_);
    if (owns_context_ && context_) aclrtDestroyContext(context_);

    coords_ = charges_ = forces_ = mesh_re_ = mesh_im_ = dft_matrix_ = influence_ = pot_ = nullptr;
    pot_host_ = host_coords_ = host_charges_ = host_dft_matrix_ = host_influence_ = nullptr;
    stream_ = nullptr;
    context_ = nullptr;
    initialized_ = false;
}
