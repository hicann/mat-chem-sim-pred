/**
 * test_vv.cpp — Velocity Verlet NPU vs CPU解析参考 精度+性能
 *
 * 解析参考: NVE下 VV积分器能量漂移对比
 * CPU解析: 同一物理公式 (半步更新 + 全步更新)
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>
#include "gaff2_types.h"
#include "gaff2_host.h"
#include "vv_host.h"
#include "vv_types.h"
#include "velocity_verlet.h"
#include "force_evaluator.h"
#include "gaff2_force_evaluator.h"

static int g_pass=0,g_fail=0;
#define CHECK(c,m) do{if(c){printf("  [PASS] %s\n",m);g_pass++;}else{printf("  [FAIL] %s\n",m);g_fail++;}}while(0)
#define CHECK_NEAR(v,r,t,m) CHECK(fabs((v)-(r))<(t),m)

// CPU reference: VV integrator one step
// v(t+dt/2) = v(t) + 0.5*dt*F(t)/m
// x(t+dt) = x(t) + dt*v(t+dt/2)
// F(t+dt) = compute_forces(x(t+dt))
// v(t+dt) = v(t+dt/2) + 0.5*dt*F(t+dt)/m
static void cpu_vv_step(double* x, double* v, double* f, const double* m, int n, double dt) {
    // half-step velocity
    for(int i=0;i<n;i++){double im=dt*0.5/m[i];v[i*3]+=f[i*3]*im;v[i*3+1]+=f[i*3+1]*im;v[i*3+2]+=f[i*3+2]*im;}
    // full-step position
    for(int i=0;i<n;i++){x[i*3]+=v[i*3]*dt;x[i*3+1]+=v[i*3+1]*dt;x[i*3+2]+=v[i*3+2]*dt;}
    // forces are recomputed externally
    // full-step velocity (done after force recompute)
    for(int i=0;i<n;i++){double im=dt*0.5/m[i];v[i*3]+=f[i*3]*im;v[i*3+1]+=f[i*3+1]*im;v[i*3+2]+=f[i*3+2]*im;}
}

int main(){
    printf("========================================\n");
    printf("  VV Integrator NPU vs CPU — Prec & Perf\n");
    printf("========================================\n\n");

    // Setup: GAFF2 + VV for 2-atom bonded system
    double masses[2]={12,12};
    double dt=0.001;

    // ── 1. Init ──
    printf("── 1. Init ──\n");
    { GAFF2Host ga; CHECK(ga.Initialize(2,1,1,0,0,10,5,0.5f,0.83333f,0,1)==0,"GAFF2");ga.Finalize(); }

    // ── 2. NVE energy drift: NPU VVIntegrator vs analytical ──
    printf("\n── 2. NVE能量守恒 (100步) ──\n");
    {
        // Two identical setups: NPU and CPU
        auto run_npu = [&](double* x, double* v, double* f, int steps, double* E0_out, double* Ef_out) -> double {
            GAFF2Host ga;
            ga.Initialize(2,1,1,0,0,10,5,0.5f,0.83333f,0,1);
            float bp[4]={1000,0.15f}; ga.UploadBondParams(bp,1);
            float tp[3]={0.3f,0,0}; ga.UploadTypeParams(tp,1);
            int32_t bonds[6]={1,0,0,0,1,0}; ga.UploadBonds(bonds,1);
            int32_t types[2]={0,0}; ga.SetAtomTypes(types);
            float cf[6]={(float)x[0],(float)x[1],(float)x[2],(float)x[3],(float)x[4],(float)x[5]};
            ga.SetCoordinates(cf);
            Gaff2ForceEvaluator fe(&ga);
            VVIntegrator vv; vv.Initialize(&fe,2,dt,masses);
            *f = ga.ComputeForces();
            *E0_out = *f + vv.ComputeKineticEnergy(v);
            for(int s=0;s<steps;s++) vv.Step(x,v,f);
            *Ef_out = *f + vv.ComputeKineticEnergy(v);
            ga.Finalize();
            return fabs(*Ef_out - *E0_out)/fabs(*E0_out);
        };

        double x_npu[6]={0,0,0,0.16,0,0}, v_npu[6]={1,0,0,-1,0,0}, f_npu[6], E0,Ef;
        double drift_npu = run_npu(x_npu,v_npu,f_npu,100,&E0,&Ef);
        CHECK_NEAR(drift_npu,0.0,1e-3,"NVE ΔE/E < 0.1% (NPU)");
        printf("    E0=%.6f E100=%.6f drift=%.2e\n",E0,Ef,drift_npu);

        // Compare final positions with CPU VV step
        double x_cpu[6]={0,0,0,0.16,0,0}, v_cpu[6]={1,0,0,-1,0,0}, f_cpu[6]={0};
        // Recompute forces for CPU ref
        GAFF2Host ga2;
        ga2.Initialize(2,1,1,0,0,10,5,0.5f,0.83333f,0,1);
        float bp2[4]={1000,0.15f}; ga2.UploadBondParams(bp2,1);
        float tp2[3]={0.3f,0,0}; ga2.UploadTypeParams(tp2,1);
        int32_t bonds2[6]={1,0,0,0,1,0}; ga2.UploadBonds(bonds2,1);
        int32_t types2[2]={0,0}; ga2.SetAtomTypes(types2);
        float cf2[6]={(float)x_cpu[0],(float)x_cpu[1],(float)x_cpu[2],(float)x_cpu[3],(float)x_cpu[4],(float)x_cpu[5]};
        ga2.SetCoordinates(cf2);
        float pot=ga2.ComputeForces(); f_cpu[0]=pot;
        for(int s=0;s<100;s++){cpu_vv_step(x_cpu,v_cpu,f_cpu,masses,2,dt);
            cf2[0]=(float)x_cpu[0];cf2[1]=(float)x_cpu[1];cf2[2]=(float)x_cpu[2];
            cf2[3]=(float)x_cpu[3];cf2[4]=(float)x_cpu[4];cf2[5]=(float)x_cpu[5];
            ga2.SetCoordinates(cf2); ga2.ComputeForces();
            const float* ff=ga2.GetForces();
            f_cpu[0]=ff[0];f_cpu[1]=ff[1];f_cpu[2]=ff[2];f_cpu[3]=ff[3];f_cpu[4]=ff[4];f_cpu[5]=ff[5];
        }
        ga2.Finalize();

        double max_dx=0;
        for(int i=0;i<6;i++){double d=fabs(x_npu[i]-x_cpu[i]); if(d>max_dx) max_dx=d;}
        CHECK_NEAR(max_dx,0.0,1e-3,"NPU vs CPU position < 1e-3 after 100 steps");
        printf("    max|Δx|_NPU-CPU=%.2e\n",max_dx);
    }

    // ── 3. Perf ──
    printf("\n── 3. 性能基准 (2原子NVE, 100步) ──\n");
    {
        GAFF2Host ga; ga.Initialize(2,1,1,0,0,10,5,0.5f,0.83333f,0,1);
        float bp[4]={1000,0.15f}; ga.UploadBondParams(bp,1);
        float tp[3]={0.3f,0,0}; ga.UploadTypeParams(tp,1);
        int32_t bonds[6]={1,0,0,0,1,0}; ga.UploadBonds(bonds,1); int32_t types[2]={0,0}; ga.SetAtomTypes(types);
        float cf[6]={0,0,0,0.16f,0,0}; ga.SetCoordinates(cf);
        Gaff2ForceEvaluator fe(&ga);
        double x[6]={0,0,0,0.16,0,0}, v[6]={1,0,0,-1,0,0}, f[6]={0};
        VVIntegrator vv; vv.Initialize(&fe,2,dt,masses);
        double dummy; vv.Step(x,v,f,&dummy);
        auto t0=std::chrono::high_resolution_clock::now();
        const int N=100; for(int i=0;i<N;i++) vv.Step(x,v,f,&dummy);
        auto t1=std::chrono::high_resolution_clock::now();
        double us=std::chrono::duration<double,std::micro>(t1-t0).count()/N;
        printf("    avg: %.1f μs/step over %d steps\n",us,N);
        CHECK(us<1e6,"Perf < 1000 ms");
        ga.Finalize();
    }

    printf("\n========================================\n");
    printf("  VV: %d PASS / %d FAIL / %d TOTAL\n",g_pass,g_fail,g_pass+g_fail);
    printf("========================================\n");
    return g_fail?1:0;
}
