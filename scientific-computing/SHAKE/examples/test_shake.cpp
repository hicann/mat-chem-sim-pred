/**
 * test_shake.cpp — SHAKE键长约束 NPU vs CPU 精度+性能
 *
 * CPU参考: 解析SHAKE迭代 (SETTLE简化版)
 * 精度: NPU约束后键长 vs CPU SHAKE后键长
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "shake.h"

static int g_pass=0, g_fail=0;
#define CHECK(c,m) do{if(c){printf("  [PASS] %s\n",m);g_pass++;}else{printf("  [FAIL] %s\n",m);g_fail++;}}while(0)
#define CHECK_NEAR(v,r,t,m) CHECK(fabs((v)-(r))<(t),m)

static double blen(const double* c,int a,int b){
    double dx=c[a*3]-c[b*3], dy=c[a*3+1]-c[b*3+1], dz=c[a*3+2]-c[b*3+2];
    return sqrt(dx*dx+dy*dy+dz*dz);
}

// CPU SHAKE: iterative Lagrange multiplier solver
static int cpu_shake(double* pos, int n, const int32_t* cons, int nc,
                     const double* target, const double* mass, double tol, int maxiter) {
    for (int iter=0; iter<maxiter; iter++) {
        double max_err=0;
        for (int c=0; c<nc; c++) {
            int a=cons[c*2], b=cons[c*2+1];
            double dx=pos[a*3]-pos[b*3], dy=pos[a*3+1]-pos[b*3+1], dz=pos[a*3+2]-pos[b*3+2];
            double r2=dx*dx+dy*dy+dz*dz, r=sqrt(r2), dr=r-target[c];
            double mi=1.0/mass[a], mj=1.0/mass[b];
            double g = dr/((mi+mj)*r);
            pos[a*3]-=g*mi*dx; pos[a*3+1]-=g*mi*dy; pos[a*3+2]-=g*mi*dz;
            pos[b*3]+=g*mj*dx; pos[b*3+1]+=g*mj*dy; pos[b*3+2]+=g*mj*dz;
            if (fabs(dr)>max_err) max_err=fabs(dr);
        }
        if (max_err<tol) return iter+1;
    }
    return maxiter;
}

int main(){
    printf("========================================\n");
    printf("  SHAKE NPU vs CPU — Precision & Perf\n");
    printf("========================================\n\n");

    // ── 1. Config ──
    printf("── 1. 配置 ──\n");
    { SHAKEConfig cfg; cfg.n_atoms=2;cfg.n_constraints=1;
      int32_t c[2]={0,1}; double t[1]={0.15}, m[2]={12,12};
      cfg.constraints=c; cfg.target_lengths=t; cfg.masses=m;
      SHAKESolver sh; CHECK(sh.Configure(cfg)==0,"Configure"); }

    // ── 2. Stretched bond: NPU vs CPU ──
    printf("\n── 2. 拉伸键修正 (r0=0.15→stretched=0.20) ──\n");
    {
        SHAKEConfig cfg; cfg.n_atoms=2; cfg.n_constraints=1; cfg.tolerance=1e-6; cfg.max_iterations=100;
        int32_t c[2]={0,1}; double t[1]={0.15}, m[2]={12,12};
        cfg.constraints=c; cfg.target_lengths=t; cfg.masses=m;

        double npu_pos[6]={0,0,0, 0.20,0,0};
        double cpu_pos[6]={0,0,0, 0.20,0,0};

        SHAKESolver sh; sh.Configure(cfg); sh.Apply(npu_pos);
        int cpu_iters = cpu_shake(cpu_pos,2,c,1,t,m,1e-6,100);

        double r_npu=blen(npu_pos,0,1), r_cpu=blen(cpu_pos,0,1);
        CHECK_NEAR(r_npu,0.15,1e-4,"NPU SHAKE restores r0");
        CHECK_NEAR(r_npu,r_cpu,1e-6,"NPU vs CPU bond length < 1e-6");
        printf("    r_NPU=%.8f r_CPU=%.8f  Δ=%.2e  CPU_iters=%d\n", r_npu, r_cpu, fabs(r_npu-r_cpu), cpu_iters);
        printf("    pos_NPU=(%.4f,%.4f,%.4f)-(%.4f,%.4f,%.4f)\n",
               npu_pos[0],npu_pos[1],npu_pos[2],npu_pos[3],npu_pos[4],npu_pos[5]);
        printf("    pos_CPU=(%.4f,%.4f,%.4f)-(%.4f,%.4f,%.4f)\n",
               cpu_pos[0],cpu_pos[1],cpu_pos[2],cpu_pos[3],cpu_pos[4],cpu_pos[5]);
    }

    // ── 3. Satisfied constraint ──
    printf("\n── 3. 已满足约束 (r=r0) ──\n");
    {
        SHAKEConfig cfg; cfg.n_atoms=2;cfg.n_constraints=1;cfg.tolerance=1e-8;cfg.max_iterations=50;
        int32_t c[2]={0,1}; double t[1]={0.15}, m[2]={1,1};
        cfg.constraints=c;cfg.target_lengths=t;cfg.masses=m;
        double pos[6]={0,0,0, 0.15,0,0};
        SHAKESolver sh; sh.Configure(cfg); sh.Apply(pos);
        CHECK_NEAR(blen(pos,0,1),0.15,1e-8,"Satisfied constraint unchanged");
    }

    // ── 4. Perf ──
    printf("\n── 4. 性能基准 (2原子, 100次) ──\n");
    {
        SHAKEConfig cfg; cfg.n_atoms=2;cfg.n_constraints=1;cfg.tolerance=1e-4;cfg.max_iterations=100;
        int32_t c[2]={0,1}; double t[1]={0.15},m[2]={12,12};
        cfg.constraints=c;cfg.target_lengths=t;cfg.masses=m;
        {double x[6]={0,0,0,0.17,0,0};SHAKESolver sh;sh.Configure(cfg);sh.Apply(x);}
        auto t0=std::chrono::high_resolution_clock::now();
        const int N=100;
        for(int i=0;i<N;i++){double x[6]={0,0,0,0.16+0.001*i,0,0};SHAKESolver sh;sh.Configure(cfg);sh.Apply(x);}
        auto t1=std::chrono::high_resolution_clock::now();
        double us=std::chrono::duration<double,std::micro>(t1-t0).count()/N;
        printf("    avg: %.1f μs over %d runs\n",us,N);
        CHECK(us<1000,"Perf < 1000 μs");
    }

    printf("\n========================================\n");
    printf("  SHAKE: %d PASS / %d FAIL / %d TOTAL\n",g_pass,g_fail,g_pass+g_fail);
    printf("========================================\n");
    return g_fail?1:0;
}
