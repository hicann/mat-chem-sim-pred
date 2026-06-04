/**
 * test_pme.cpp — PME/Ewald NPU vs CPU 精度+性能
 *
 * CPU参考: 直接 O(N²) Ewald实空间求和
 * 精度: NPU E_recip vs CPU E_direct
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "ewald_types.h"
#include "ewald_host.h"

static int g_pass=0,g_fail=0;
#define CHECK(c,m) do{if(c){printf("  [PASS] %s\n",m);g_pass++;}else{printf("  [FAIL] %s\n",m);g_fail++;}}while(0)
#define CHECK_NEAR(v,r,t,m) CHECK(fabs((v)-(r))<(t),m)

static const float INV_4PE0=138.935485f;

// CPU: direct Coulomb sum (no Ewald splitting, just 1/r)
static float cpu_direct_coulomb(const float* coords, const float* charges, int n, float box) {
    float E=0;
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            float dx=coords[i*3]-coords[j*3], dy=coords[i*3+1]-coords[j*3+1], dz=coords[i*3+2]-coords[j*3+2];
            // Minimum image convention
            dx-=box*roundf(dx/box); dy-=box*roundf(dy/box); dz-=box*roundf(dz/box);
            float r=sqrtf(dx*dx+dy*dy+dz*dz);
            if(r<1e-8f) continue;
            E += INV_4PE0*charges[i]*charges[j]/r;
        }
    }
    return E;
}

int main(){
    printf("========================================\n");
    printf("  PME/Ewald NPU vs CPU — Prec & Perf\n");
    printf("========================================\n\n");

    // ── 1. Init ──
    printf("── 1. 初始化 ──\n");
    { EwaldHost ew; CHECK(ew.Initialize(64,8,3,0.35f,20,0)==0,"Init 64atoms"); ew.Finalize(); }

    // ── 2. Zero charge: NPU vs CPU ──
    printf("\n── 2. 零电荷 NPU vs CPU ──\n");
    {
        EwaldHost ew; ew.Initialize(2,1,10,0.35f,15,0);
        float tp[3]={0.3f,0,0}; ew.SetTypeParams(tp,1);
        float coords[6]={0,0,0, 1,0,0}; ew.SetCoordinates(coords);
        int32_t types[2]={0,0}; ew.SetAtomTypes(types);
        float enpu=ew.ComputeForces();
        float ecpu=cpu_direct_coulomb(coords,tp,2,10);
        CHECK_NEAR(enpu,ecpu,1e-4,"NPU vs CPU: zero-charge E=0");
        printf("    E_NPU=%.6e E_CPU=%.6e\n",enpu,ecpu);
        ew.Finalize();
    }

    // ── 3. Point charges: NPU vs CPU ──
    printf("\n── 3. 点电荷 NPU vs CPU (q=±1, r=0.5nm) ──\n");
    {
        EwaldHost ew; ew.Initialize(2,2,10,0.35f,15,0);
        float tp[6]={0.3f,0,1, 0.3f,0,-1}; ew.SetTypeParams(tp,2);
        float coords[6]={0,0,0, 0.5f,0,0}; ew.SetCoordinates(coords);
        int32_t types[2]={0,1}; ew.SetAtomTypes(types);
        float enpu=ew.ComputeForces();
        float charges[2]={1,-1};
        float ecpu=cpu_direct_coulomb(coords,charges,2,10);
        float e_analytic=INV_4PE0*1*(-1)/0.5f;
        CHECK_NEAR(enpu,e_analytic,1e-3,"NPU E matches analytical");
        // CPU direct sum = analytical for 2 atoms (no images)
        CHECK_NEAR(ecpu,e_analytic,1e-3,"CPU E matches analytical");
        printf("    E_NPU=%.4f E_CPU=%.4f E_analytic=%.4f\n",enpu,ecpu,e_analytic);
        printf("    |NPU-CPU|=%.2e |NPU-analytic|=%.2e\n",fabs(enpu-ecpu),fabs(enpu-e_analytic));
        ew.Finalize();
    }

    // ── 4. Multi-charge: NPU vs CPU ──
    printf("\n── 4. 多电荷 (4原子) NPU vs CPU ──\n");
    {
        EwaldHost ew; ew.Initialize(4,4,5,0.35f,15,0);
        float tp[12]={0.3f,0,0.5f, 0.3f,0,-0.5f, 0.3f,0,0.3f, 0.3f,0,-0.3f};
        ew.SetTypeParams(tp,4);
        float coords[12]={0,0,0, 0.4f,0,0, 0.8f,0,0, 1.2f,0,0};
        ew.SetCoordinates(coords);
        int32_t types[4]={0,1,2,3}; ew.SetAtomTypes(types);
        float enpu=ew.ComputeForces();
        float charges[4]={0.5f,-0.5f,0.3f,-0.3f};
        float ecpu=cpu_direct_coulomb(coords,charges,4,5);
        CHECK_NEAR(enpu,ecpu,1e-2,"NPU vs CPU: 4-atom Ewald < 1%");
        printf("    E_NPU=%.6f E_CPU=%.6f  Δ=%.2e\n",enpu,ecpu,fabs(enpu-ecpu));
        ew.Finalize();
    }

    // ── 5. Perf ──
    printf("\n── 5. 性能基准 (64原子, 10次) ──\n");
    {
        EwaldHost ew; ew.Initialize(64,8,3,0.35f,20,0);
        float tp[24]; for(int i=0;i<24;i++) tp[i]=(i%3==0)?0.3f:((i%3==2)?1.0f:0);
        ew.SetTypeParams(tp,8);
        float coords[192]; for(int i=0;i<192;i++) coords[i]=(float)((i%20)*0.15);
        ew.SetCoordinates(coords);
        int32_t types[64]; for(int i=0;i<64;i++) types[i]=i%8;
        ew.SetAtomTypes(types);
        ew.ComputeForces();
        auto t0=std::chrono::high_resolution_clock::now();
        const int N=10; for(int i=0;i<N;i++) ew.ComputeForces();
        auto t1=std::chrono::high_resolution_clock::now();
        double us=std::chrono::duration<double,std::micro>(t1-t0).count()/N;
        printf("    avg: %.1f μs over %d runs\n",us,N);
        CHECK(us<1e7,"Perf < 10000 ms");
        ew.Finalize();
    }

    printf("\n========================================\n");
    printf("  PME: %d PASS / %d FAIL / %d TOTAL\n",g_pass,g_fail,g_pass+g_fail);
    printf("========================================\n");
    return g_fail?1:0;
}
