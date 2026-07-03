/**
 * vv_host.cpp
 *
 * VV Integrator + NPT Thermostat/Barostat — Host-Side Implementation
 *
 * Orchestrates the 3 NPU kernels that together form the NPT step:
 *   vv_integrate → force eval → vv_finish → host KE/virial → thermo_scale
 *
 * The host-side scalar logic replaces what was in npt_ensemble.h's CPU loops.
 * This enables the NPT step to run entirely on NPU + minimal host scalars.
 */

#include <new>

#include "vv_host.h"

// ============================================================
// CANN Runtime API declarations
// ============================================================
extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim,
                            void **args, uint32_t size, const void *stream);
int UnregisterAscendBinary(void *hdl);
}

// ============================================================
// Embedded kernel binaries
// ============================================================
static const unsigned char vv_integrate_kernel_bin[] = {
#include "vv_integrate_device_aiv_bin.inc"
};
static const uint64_t vv_integrate_kernel_len = sizeof(vv_integrate_kernel_bin);

static const unsigned char vv_finish_kernel_bin[] = {
#include "vv_finish_device_aiv_bin.inc"
};
static const uint64_t vv_finish_kernel_len = sizeof(vv_finish_kernel_bin);

static const unsigned char thermo_scale_kernel_bin[] = {
#include "thermo_scale_device_aiv_bin.inc"
};
static const uint64_t thermo_scale_kernel_len = sizeof(thermo_scale_kernel_bin);

static void* g_vv_integrate_handle = nullptr;
static bool g_vv_integrate_loaded = false;
static void* g_vv_finish_handle = nullptr;
static bool g_vv_finish_loaded = false;
static void* g_thermo_scale_handle = nullptr;
static bool g_thermo_scale_loaded = false;

// ============================================================
// Kernel registration
// ============================================================
static int32_t VVHost_InitKernel(
    const unsigned char* bin, uint64_t bin_len,
    void** handle, bool* loaded, const char* name)
{
    if (*loaded) return 0;
    uint32_t ret = RegisterAscendBinary(
        (const char*)bin, bin_len, 1, handle
    );
    if (ret != 0 || !*handle) {
        fprintf(stderr, "[VV] FATAL: RegisterAscendBinary(%s) failed (ret=%u, handle=%p)\n",
                name, ret, *handle);
        return -1;
    }
    fprintf(stdout, "[VV] Kernel registered: %s (handle=%p, size=%llu)\n",
            name, *handle, (unsigned long long)bin_len);
    *loaded = true;
    return 0;
}

// ============================================================
// Kernel launch
// ============================================================
static int32_t VVHost_LaunchKernel(
    void* handle, void* args, uint32_t argsSize, aclrtStream stream)
{
    if (!handle) return -1;

    void* overflow_dev = nullptr;
    aclrtMalloc(&overflow_dev, 8, ACL_MEM_MALLOC_HUGE_FIRST);
    if (overflow_dev) {
        uint64_t zero = 0;
        aclrtMemcpy(overflow_dev, 8, &zero, 8, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    struct {
        VVForceArgs args;
        void* __ascendc_overflow;
    } wrapper;

    memcpy(&wrapper.args, args, sizeof(VVForceArgs));
    wrapper.__ascendc_overflow = overflow_dev;

    uint32_t ret = LaunchAscendKernel(handle, 0, 1,
        (void**)&wrapper, sizeof(wrapper), stream);

    if (overflow_dev) aclrtFree(overflow_dev);
    return (int32_t)ret;
}

// ============================================================
// VVHost implementation
// ============================================================

int32_t VVHost::Initialize(
    int32_t n_atoms, double dt,
    const double* masses,
    int32_t n_dof,
    void* shared_coords,
    void* shared_forces,
    aclrtContext external_context,
    aclrtStream external_stream,
    int32_t device_id)
{
    if (initialized_) return 0;

    n_atoms_ = n_atoms;
    device_id_ = device_id;
    n_dof_ = n_dof;
    dt_ = dt;
    half_dt_ = dt * 0.5;

    int32_t ret;
    ret = VVHost_InitKernel(vv_integrate_kernel_bin, vv_integrate_kernel_len,
                            &g_vv_integrate_handle, &g_vv_integrate_loaded, "vv_integrate");
    if (ret != 0) return ret;

    ret = VVHost_InitKernel(vv_finish_kernel_bin, vv_finish_kernel_len,
                            &g_vv_finish_handle, &g_vv_finish_loaded, "vv_finish");
    if (ret != 0) return ret;

    ret = VVHost_InitKernel(thermo_scale_kernel_bin, thermo_scale_kernel_len,
                            &g_thermo_scale_handle, &g_thermo_scale_loaded, "thermo_scale");
    if (ret != 0) return ret;

    if (external_context && external_stream) {
        context_ = external_context;
        stream_ = external_stream;
    } else {
        aclrtSetDevice(device_id_);
        aclrtContext ctx = nullptr;
        aclrtCreateContext(&ctx, device_id_);
        context_ = ctx;
        aclrtCreateStream(&stream_);
    }

    coords_ = shared_coords;
    forces_ = shared_forces;

    // CWE-476/190 fix: compute sizes in 64-bit and check ALL aclrtMalloc returns.
    // Previously velocities_/masses_/pot_virial_ were allocated without checking
    // aclrtMalloc's return, so a failure left them null and later used (CWE-476).
    // Sizes are also computed in 64-bit to avoid int32 overflow (CWE-190).
    if (n_atoms_ <= 0) {
        fprintf(stderr, "[VV] Initialize: invalid n_atoms=%d\n", n_atoms_);
        return -1;
    }
    size_t vel_sz = static_cast<size_t>(n_atoms_) * 3 * sizeof(float);
    size_t mass_sz = static_cast<size_t>(n_atoms_) * sizeof(float);
    size_t pv_sz = 4 * sizeof(float);

    if (aclrtMalloc(&velocities_, vel_sz, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        fprintf(stderr, "[VV] aclrtMalloc velocities_ failed\n");
        return -1;
    }
    if (aclrtMalloc(&masses_, mass_sz, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        fprintf(stderr, "[VV] aclrtMalloc masses_ failed\n");
        aclrtFree(velocities_); velocities_ = nullptr;
        return -1;
    }
    if (aclrtMalloc(&pot_virial_, pv_sz, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        fprintf(stderr, "[VV] aclrtMalloc pot_virial_ failed\n");
        aclrtFree(masses_); masses_ = nullptr;
        aclrtFree(velocities_); velocities_ = nullptr;
        return -1;
    }

    float zeros[4] = {0};
    aclrtMemcpy(pot_virial_, pv_sz, zeros, pv_sz, ACL_MEMCPY_HOST_TO_DEVICE);
    pot_virial_host_ = (float*)malloc(pv_sz);

    masses_float_.resize(n_atoms_);
    for (int32_t i = 0; i < n_atoms_; i++) {
        float m = masses ? (float)masses[i] : 1.0f;
        masses_float_[i] = m;
    }
    aclrtMemcpy(masses_, mass_sz, masses_float_.data(), mass_sz, ACL_MEMCPY_HOST_TO_DEVICE);

    rng_.seed(42);
    initialized_ = true;
    return 0;
}

// ============================================================
// StepIntegrate: vv_integrate kernel only
// v(t+dt/2) = v(t) + (dt/2) * F(t) / m
// r(t+dt)   = r(t) + dt * v(t+dt/2)
// ============================================================
int32_t VVHost::StepIntegrate() {
    VVConfig config;
    memset(&config, 0, sizeof(config));
    config.n_atoms = n_atoms_;
    config.dt = (float)dt_;
    config.half_dt = (float)half_dt_;
    config.box_size = 10.0f;  // box_size is not used for the vv_integrate PBC
    config.half_box = 5.0f;   // (vv_integrate uses config for box/half_box for PBC)

    VVKernelGM gm;
    gm.coords_gm = coords_;
    gm.velocities_gm = velocities_;
    gm.forces_gm = forces_;
    gm.masses_gm = masses_;
    gm.pot_virial_gm = pot_virial_;

    VVForceArgs args;
    args.config = config;
    args.gm = gm;
    args.tile_start = 0;
    args.tile_end = n_atoms_;

    return VVHost_LaunchKernel(g_vv_integrate_handle, &args, sizeof(args), stream_);
}

// ============================================================
// StepFinish: vv_finish kernel + host-side scalar calc
// v(t+dt) = v(t+dt/2) + (dt/2) * F(t+dt) / m
// KE += 0.5 * m * |v|²
// virial += r · F
// Host: compute V-rescale λ and C-rescale μ
// ============================================================
void VVHost::StepFinish(
    double* temperature_out,
    double* virial_out,
    double* box_size)
{
    // Initialize pot_virial to zero on device
    float zeros[4] = {0};
    aclrtMemcpy(pot_virial_, 4 * sizeof(float), zeros, 4 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    VVConfig config;
    memset(&config, 0, sizeof(config));
    config.n_atoms = n_atoms_;
    config.dt = (float)dt_;
    config.half_dt = (float)half_dt_;

    VVKernelGM gm;
    gm.coords_gm = coords_;
    gm.velocities_gm = velocities_;
    gm.forces_gm = forces_;
    gm.masses_gm = masses_;
    gm.pot_virial_gm = pot_virial_;

    VVForceArgs args;
    args.config = config;
    args.gm = gm;
    args.tile_start = 0;
    args.tile_end = n_atoms_;

    int32_t ret = VVHost_LaunchKernel(g_vv_finish_handle, &args, sizeof(args), stream_);
    if (ret != 0) {
        fprintf(stderr, "[VV] vv_finish kernel launch failed\n");
        if (temperature_out) *temperature_out = 0;
        if (virial_out) *virial_out = 0;
        vrescale_lambda_ = 1.0;
        crescale_mu_ = 1.0;
        v_scale_ = 1.0;
        return;
    }

    // Read KE and virial from device
    aclrtSynchronizeStream(stream_);
    aclrtMemcpy(pot_virial_host_, 4 * sizeof(float),
                pot_virial_, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    double ke = (double)pot_virial_host_[0];
    double virial = (double)pot_virial_host_[1];
    double temperature = 2.0 * ke / (n_dof_ * KB_VVH);

    if (temperature_out) *temperature_out = temperature;
    if (virial_out) *virial_out = virial;

    // Host: V-rescale λ computation
    vrescale_lambda_ = 1.0;
    if (cfg_tau_t_ > 0 && ke > 1e-30) {
        double ke_target = 0.5 * n_dof_ * KB_VVH * cfg_target_temp_;
        double df = std::exp(-dt_ / cfg_tau_t_);
        double ke_ratio = ke_target / ke;
        double xi = BoxMuller();
        double lambda_sq = 1.0 + (1.0 - df) * (ke_ratio - 1.0)
                         + std::sqrt(ke_ratio) * std::sqrt(1.0 - df * df) * xi;
        if (lambda_sq <= 0) lambda_sq = 1e-10;
        vrescale_lambda_ = std::sqrt(lambda_sq);
    }

    // Host: C-rescale μ computation
    crescale_mu_ = 1.0;
    v_scale_ = 1.0;
    if (cfg_tau_p_ > 0 && box_size) {
        double volume = (*box_size) * (*box_size) * (*box_size);
        double pressure = (2.0 * ke + virial) / (3.0 * volume);
        double exponent = -dt_ / (3.0 * cfg_tau_p_) * cfg_compressibility_ * (pressure - cfg_target_pressure_);
        crescale_mu_ = std::exp(exponent);
        const double MAX_SCALE = 1.01, MIN_SCALE = 0.99;
        if (crescale_mu_ > MAX_SCALE) crescale_mu_ = MAX_SCALE;
        if (crescale_mu_ < MIN_SCALE) crescale_mu_ = MIN_SCALE;
        v_scale_ = 1.0 / std::cbrt(crescale_mu_);
        *box_size *= crescale_mu_;
    }
}

// ============================================================
// StepVRescale: Apply V-rescale lambda (host-side velocity scaling)
// ============================================================
int32_t VVHost::StepVRescale() {
    if (cfg_tau_t_ <= 0) return 0;
    // Download velocities, scale on host, upload back.
    // CWE-190/alloc fix: compute element/byte counts in 64-bit to avoid int32
    // overflow (n_atoms_*3 overflows when n_atoms_ > ~715M), and use nothrow new
    // with a null check instead of throwing new[] (bad_alloc) which would
    // terminate the whole MD simulation on an allocation failure.
    size_t n_vel = static_cast<size_t>(n_atoms_) * 3;
    size_t vel_bytes = n_vel * sizeof(float);
    float* vel = new (std::nothrow) float[n_vel];
    if (!vel) {
        fprintf(stderr, "[VV] StepVRescale: failed to allocate velocity buffer\n");
        return -1;
    }
    aclrtMemcpy(vel, vel_bytes, velocities_, vel_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    float lambda_f = (float)vrescale_lambda_;
    for (size_t i = 0; i < n_vel; i++) {
        vel[i] *= lambda_f;
    }
    aclrtMemcpy(velocities_, vel_bytes, vel, vel_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    delete[] vel;
    return 0;
}

// ============================================================
// StepScale: thermo_scale kernel (apply V-rescale + C-rescale)
// ============================================================
int32_t VVHost::StepScale() {
    VVConfig config;
    memset(&config, 0, sizeof(config));
    config.n_atoms = n_atoms_;
    config.dt = (float)dt_;
    config.half_dt = (float)half_dt_;
    config.tau_t = (float)cfg_tau_t_;
    config.tau_p = (float)cfg_tau_p_;
    config.enable_vrescale = (cfg_tau_t_ > 0) ? 1 : 0;
    config.enable_barostat = (cfg_tau_p_ > 0) ? 1 : 0;
    config.vrescale_lambda = (float)vrescale_lambda_;
    config.crescale_mu = (float)crescale_mu_;
    config.v_scale = (float)v_scale_;

    VVKernelGM gm;
    gm.coords_gm = coords_;
    gm.velocities_gm = velocities_;
    gm.forces_gm = forces_;
    gm.masses_gm = masses_;
    gm.pot_virial_gm = pot_virial_;

    VVForceArgs args;
    args.config = config;
    args.gm = gm;
    args.tile_start = 0;
    args.tile_end = n_atoms_;

    return VVHost_LaunchKernel(g_thermo_scale_handle, &args, sizeof(args), stream_);
}

// ============================================================
// Original Step — calls all three split steps in sequence
// ============================================================
double VVHost::Step(
    double* temperature_out,
    double* virial_out,
    double* box_size,
    double target_temp,
    double target_pressure,
    double tau_t,
    double tau_p,
    double collision_freq,
    double compressibility)
{
    if (!initialized_) return 0.0;

    // Set parameters and call split steps
    SetNPTParams(dt_, target_temp, target_pressure, tau_t, tau_p,
                 collision_freq, compressibility);

    int32_t ret = StepIntegrate();
    if (ret != 0) {
        fprintf(stderr, "[VV] StepIntegrate failed\n");
        return 0.0;
    }

    // NOTE: Forces must be evaluated after StepIntegrate updates coordinates!
    // This must be done by the caller before calling StepFinish.

    // For backward compatibility: StepFinish + StepVRescale + StepScale
    double temperature = 0, virial = 0;
    double box = box_size ? *box_size : 10.0;
    StepFinish(&temperature, &virial, &box);

    // 检查 StepVRescale 返回值：若速度缩放分配失败(返回 -1)却继续执行
    // StepScale()，会用未经 λ 缩放的错误速度数据送入 thermo_scale 内核，
    // 产生静默数值错误(无报错无崩溃)，比旧代码 bad_alloc 终止更危险。
    // 失败时与上方 StepIntegrate 错误处理一致，直接返回 0.0 终止本步。
    if (cfg_tau_t_ > 0.0) {
        int32_t vret = StepVRescale();
        if (vret != 0) {
            fprintf(stderr, "[VV] StepVRescale failed (ret=%d)\n", vret);
            return 0.0;
        }
    }

    StepScale();

    if (temperature_out) *temperature_out = temperature;
    if (virial_out) *virial_out = virial;
    if (box_size) *box_size = box;

    return 0.5 * n_dof_ * KB_VVH * temperature;  // KE = 0.5 * n_dof * kT
}

// ============================================================
// Box-Muller Gaussian RNG
// ============================================================
double VVHost::BoxMuller() {
    double u1 = uniform_dist_(rng_);
    double u2 = uniform_dist_(rng_);
    return std::sqrt(-2.0 * std::log(u1 + 1e-30)) * std::cos(2.0 * M_PI * u2);
}

// ============================================================
// Cleanup
// ============================================================
void VVHost::Finalize() {
    if (!initialized_) return;
    if (velocities_) aclrtFree(velocities_);
    if (masses_) aclrtFree(masses_);
    if (pot_virial_) aclrtFree(pot_virial_);
    if (pot_virial_host_) free(pot_virial_host_);
    coords_ = velocities_ = forces_ = masses_ = pot_virial_ = nullptr;
    pot_virial_host_ = nullptr;
    initialized_ = false;
}
