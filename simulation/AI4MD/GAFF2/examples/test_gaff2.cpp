/**
 * test_gaff2.cpp — GAFF2力场 NPU算子 精度(NPU vs CPU) + 性能 单元测试
 *
 * CPU参考: 标准C++数学库实现 Harmonic Bond + LJ-12-6 + Coulomb
 * NPU vs CPU: 比较力向量、能量, 输出最大相对误差
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>
#include "gaff2_types.h"
#include "gaff2_host.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); g_pass++; } \
    else { printf("  [FAIL] %s\n", msg); g_fail++; } \
} while(0)
#define CHECK_NEAR(val, ref, tol, msg) \
    CHECK(fabs((val)-(ref)) < (tol), msg)

// ═══════════════════════════════════════════════════════════════
// CPU Reference: GAFF2 force computation (standard C++ math)
// ═══════════════════════════════════════════════════════════════

// Bond: harmonic E=½k(r-r0)², F_i=k(r-r0)*dr_i/r (signed)
static void cpu_bond_force(const float* coords, const int32_t* bonds, int nb,
                            const float* bond_params, int nbp, float* f, float* e) {
    for (int b = 0; b < nb; b++) {
        int tp = bonds[b*6];
        int a  = bonds[b*6+3], bb = bonds[b*6+4]; // 1-indexed, assuming already 0-indexed from test
        float k = bond_params[tp*4], r0 = bond_params[tp*4+1];
        float dx=coords[a*3]-coords[bb*3], dy=coords[a*3+1]-coords[bb*3+1], dz=coords[a*3+2]-coords[bb*3+2];
        float r = sqrtf(dx*dx+dy*dy+dz*dz);
        if (r < 1e-8f) continue;
        float dr = r - r0;
        float fmag = -2.0f * k * dr;  // F = -dE/dr, sign convention: toward equilibrium
        float fx = fmag * dx / r, fy = fmag * dy / r, fz = fmag * dz / r;
        f[a*3]+=fx; f[a*3+1]+=fy; f[a*3+2]+=fz;
        f[bb*3]-=fx; f[bb*3+1]-=fy; f[bb*3+2]-=fz;
        *e += 0.5f * k * dr * dr;
    }
}

// LJ-12-6: E=4ε[(σ/r)¹²-(σ/r)⁶], F=24ε/r·[2(σ/r)¹²-(σ/r)⁶]
static void cpu_lj_force(const float* coords, const int32_t* types, int n,
                          const float* type_params, int ntp, float cutoff,
                          float* f, float* enb) {
    float csq = cutoff * cutoff;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            float dx=coords[i*3]-coords[j*3], dy=coords[i*3+1]-coords[j*3+1], dz=coords[i*3+2]-coords[j*3+2];
            float r2 = dx*dx+dy*dy+dz*dz;
            if (r2 > csq || r2 < 1e-12f) continue;
            float r = sqrtf(r2), inv_r = 1.0f/r;
            int ti = types[i], tj = types[j];
            float si = type_params[ti*3], ei = type_params[ti*3+1];
            float sj = type_params[tj*3], ej = type_params[tj*3+1];
            // Lorentz-Berthelot combining rules
            float sigma = 0.5f*(si+sj), eps = sqrtf(ei*ej);
            float sr = sigma*inv_r, sr2=sr*sr, sr6=sr2*sr2*sr2, sr12=sr6*sr6;
            float elj = 4.0f*eps*(sr12 - sr6);
            float fmag = 24.0f*eps*inv_r*(2.0f*sr12 - sr6);
            float fx=fmag*dx*inv_r, fy=fmag*dy*inv_r, fz=fmag*dz*inv_r;
            f[i*3]+=fx; f[i*3+1]+=fy; f[i*3+2]+=fz;
            f[j*3]-=fx; f[j*3+1]-=fy; f[j*3+2]-=fz;
            *enb += elj;
        }
    }
}

// Coulomb: E=q_i*q_j/(4πε₀r), F=q_i*q_j/(4πε₀r²)  (ε₀=138.935485 kJ·nm/mol/e² for GROMACS)
static const float INV_4PE0 = 138.935485f;

static void cpu_coulomb_force(const float* coords, const int32_t* types, int n,
                               const float* type_params, int ntp, float cutoff,
                               float* f, float* ec) {
    float csq = cutoff * cutoff;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            float dx=coords[i*3]-coords[j*3], dy=coords[i*3+1]-coords[j*3+1], dz=coords[i*3+2]-coords[j*3+2];
            float r2 = dx*dx+dy*dy+dz*dz;
            if (r2 > csq || r2 < 1e-12f) continue;
            float r = sqrtf(r2), inv_r = 1.0f/r;
            float qi = type_params[types[i]*3+2], qj = type_params[types[j]*3+2];
            float ec_ij = INV_4PE0 * qi * qj * inv_r;
            float fmag = ec_ij * inv_r;
            float fx=fmag*dx*inv_r, fy=fmag*dy*inv_r, fz=fmag*dz*inv_r;
            f[i*3]+=fx; f[i*3+1]+=fy; f[i*3+2]+=fz;
            f[j*3]-=fx; f[j*3+1]-=fy; f[j*3+2]-=fz;
            *ec += ec_ij;
        }
    }
}

// Compare two force arrays, return max relative error
static double compare_forces(const float* fnpu, const float* fcpu, int n,
                              const char* label, double* out_max_abs) {
    double max_rel = 0.0, max_abs = 0.0;
    for (int i = 0; i < n*3; i++) {
        double diff = fabs((double)fnpu[i] - (double)fcpu[i]);
        double ref = fmax(fabs((double)fcpu[i]), 1e-6);
        double rel = diff / ref;
        if (rel > max_rel) max_rel = rel;
        if (diff > max_abs) max_abs = diff;
    }
    *out_max_abs = max_abs;
    return max_rel;
}

// ═══════════════════════════════════════════════════════════════
int main() {
    printf("========================================\n");
    printf("  GAFF2 NPU vs CPU — Precision & Perf\n");
    printf("========================================\n\n");

    // ── 1. Init ──────────────────────────────────────────────
    printf("── 1. 初始化 ──\n");
    { GAFF2Host g; CHECK(g.Initialize(4,3,2,1,0,10,2.5f)==0,"Init"); g.Finalize(); }

    // ── 2. Bond: NPU vs CPU ──────────────────────────────────
    printf("\n── 2. Bond力 NPU vs CPU (2原子, r=r0+0.01nm) ──\n");
    {
        int n_atoms = 2;
        GAFF2Host g;
        g.Initialize(n_atoms, 1, 1, 0, 0, 10, 5, 0.5f, 0.83333f, 0, 1);
        float bp[4]={1000,0.15f}; g.UploadBondParams(bp,1);
        float tp[3]={0.3f,0,0}; g.UploadTypeParams(tp,1);
        int32_t bonds[6]={1,0,0, 0,1,0}; g.UploadBonds(bonds,1);
        int32_t types[2]={0,0}; g.SetAtomTypes(types);
        float coords[6]={0,0,0, 0.16f,0,0}; g.SetCoordinates(coords);

        float enpu = g.ComputeForces();
        const float* fnpu = g.GetForces();

        // CPU ref
        float fcpu[6]={0}, ecpu_bond=0;
        cpu_bond_force(coords, bonds, 1, bp, 1, fcpu, &ecpu_bond);

        double max_abs;
        double max_rel = compare_forces(fnpu, fcpu, n_atoms, "Bond", &max_abs);
        CHECK(max_rel < 1e-4, "Bond force: max_rel < 1e-4 vs CPU");
        CHECK_NEAR(enpu, ecpu_bond, 1e-4f, "Bond energy: NPU vs CPU < 1e-4");
        printf("    NPU E=%.6f  CPU E=%.6f  max|ΔF|=%.2e  max_rel=%.2e\n", enpu, ecpu_bond, max_abs, max_rel);
        g.Finalize();
    }

    // ── 3. LJ: NPU vs CPU ────────────────────────────────────
    printf("\n── 3. LJ NPU vs CPU (2原子, r=r_min) ──\n");
    {
        int n_atoms = 2;
        GAFF2Host g;
        g.Initialize(n_atoms, 1, 0, 0, 0, 10, 10, 0.5f, 0.83333f, 0, 1);
        float tp[3]={0.3f,1.0f,0}; g.UploadTypeParams(tp,1);
        int32_t types[2]={0,0}; g.SetAtomTypes(types);
        float r_min = 0.3f*powf(2,1.0f/6);
        float coords[6]={0,0,0, r_min,0,0}; g.SetCoordinates(coords);

        float enpu = g.ComputeForces();
        const float* fnpu = g.GetForces();

        float fcpu[6]={0}, enb_cpu=0;
        cpu_lj_force(coords, types, n_atoms, tp, 1, 10, fcpu, &enb_cpu);

        double max_abs;
        double max_rel = compare_forces(fnpu, fcpu, n_atoms, "LJ", &max_abs);
        CHECK(max_rel < 1e-3, "LJ force: max_rel < 1e-3 vs CPU");
        CHECK_NEAR(enpu, enb_cpu, 1e-3f, "LJ energy: NPU vs CPU < 1e-3");
        printf("    NPU E=%.6f  CPU E=%.6f  max|ΔF|=%.2e  max_rel=%.2e\n", enpu, enb_cpu, max_abs, max_rel);
        g.Finalize();
    }

    // ── 4. Coulomb: NPU vs CPU ──────────────────────────────
    printf("\n── 4. Coulomb NPU vs CPU (2原子, q=±1e) ──\n");
    {
        int n_atoms = 2;
        GAFF2Host g;
        g.Initialize(n_atoms, 2, 0, 0, 0, 10, 5, 0.5f, 0.83333f, 0, 1);
        float tp[6]={0.3f,0,1.0f, 0.3f,0,-1.0f}; g.UploadTypeParams(tp,2);
        int32_t types[2]={0,1}; g.SetAtomTypes(types);
        float coords[6]={0,0,0, 0.5f,0,0}; g.SetCoordinates(coords);

        float enpu = g.ComputeForces();
        const float* fnpu = g.GetForces();

        float fcpu[6]={0}, ec_cpu=0;
        cpu_coulomb_force(coords, types, n_atoms, tp, 2, 5, fcpu, &ec_cpu);

        double max_abs;
        double max_rel = compare_forces(fnpu, fcpu, n_atoms, "Coulomb", &max_abs);
        CHECK(max_rel < 1e-3, "Coulomb force: max_rel < 1e-3 vs CPU");
        // E_c = 138.935 * (+1)*(-1) / 0.5 = -277.87 kJ/mol
        float ec_expected = INV_4PE0 * 1.0f * (-1.0f) / 0.5f;
        CHECK_NEAR(enpu, ec_expected, 1e-3f, "Coulomb E matches analytical");
        printf("    NPU E=%.4f  CPU E=%.4f  analytical E=%.4f  max_rel=%.2e\n", enpu, ec_cpu, ec_expected, max_rel);
        g.Finalize();
    }

    // ── 5. Performance ──────────────────────────────────────
    printf("\n── 5. 性能基准 (64原子, 10次) ──\n");
    {
        GAFF2Host g;
        g.Initialize(64,8,63,0,0,3,1.2f,0.5f,0.83333f,0,1);
        float tp[24]; for(int i=0;i<24;i++) tp[i]=(i%3==0)?0.3f:((i%3==1)?1.0f:0);
        g.UploadTypeParams(tp,8);
        float bp[12]={500,0.15f,0,0, 500,0.12f,0,0, 500,0.14f,0,0};
        g.UploadBondParams(bp,3);
        int32_t bonds[378]; for(int i=0;i<63;i++){bonds[i*6]=i%3;bonds[i*6+3]=i;bonds[i*6+4]=i+1;}
        g.UploadBonds(bonds,63);
        float coords[192]; for(int i=0;i<192;i++) coords[i]=(float)((i%30)*0.1);
        g.SetCoordinates(coords);
        int32_t types[64]; for(int i=0;i<64;i++) types[i]=i%8;
        g.SetAtomTypes(types);
        g.ComputeForces(); // warmup
        auto t0=std::chrono::high_resolution_clock::now();
        const int N=10; for(int i=0;i<N;i++) g.ComputeForces();
        auto t1=std::chrono::high_resolution_clock::now();
        double us=std::chrono::duration<double,std::micro>(t1-t0).count()/N;
        printf("    avg: %.1f μs (%.2f ms) over %d runs\n", us, us/1000, N);
        CHECK(us<5e6, "Perf < 5000 ms");
        g.Finalize();
    }

    printf("\n========================================\n");
    printf("  GAFF2: %d PASS / %d FAIL / %d TOTAL\n", g_pass, g_fail, g_pass+g_fail);
    printf("========================================\n");
    return g_fail ? 1 : 0;
}
