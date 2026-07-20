#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "acl/acl.h"
#include "aclnn_sru_scan_fused.h"

#define CHECK_ACL(expr) do { auto _ret = (expr); if (_ret != ACL_SUCCESS) { std::cerr << #expr << " failed, ret=" << _ret << std::endl; return 1; } } while (false)
#define CHECK_ACLNN(expr) do { auto _ret = (expr); if (_ret != 0) { std::cerr << #expr << " failed, ret=" << _ret << std::endl; return 1; } } while (false)

namespace {

aclTensor* MakeTensor(const std::vector<int64_t>& dims, aclDataType dtype, void* data) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int64_t i = (int64_t)dims.size() - 2; i >= 0; --i) strides[i] = strides[i+1] * dims[i+1];
  return aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, dims.data(), dims.size(), data);
}

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// CPU oracle: SRU
void Ref(int64_t B, int64_t L, int64_t IN, int64_t H,
         const std::vector<float>& x, const std::vector<float>& wt,
         const std::vector<float>& bias, std::vector<float>& out) {
  std::vector<float> c(H, 0.0f), xtv(H), fv(H), rv(H);
  for (int64_t bb = 0; bb < B; ++bb) {
    std::fill(c.begin(), c.end(), 0.0f);
    for (int64_t s = 0; s < L; ++s) {
      for (int64_t m = 0; m < H; ++m) {
        float ax=0, af=0, ar=0;
        for (int64_t k = 0; k < IN; ++k) {
          float xv = x[((bb*L)+s)*IN+k];
          ax += xv * wt[k*H+m];
          af += xv * wt[(IN+k)*H+m];
          ar += xv * wt[(2*IN+k)*H+m];
        }
        xtv[m] = ax;
        fv[m] = Sigmoid(af + bias[m]*c[m] + bias[2*H+m]);
        rv[m] = Sigmoid(ar + bias[H+m]*c[m] + bias[3*H+m]);
      }
      for (int64_t m = 0; m < H; ++m) {
        c[m] = fv[m]*c[m] + (1.0f-fv[m])*xtv[m];
        float h = rv[m]*std::tanh(c[m]) + (1.0f-rv[m])*xtv[m];
        out[((bb*L)+s)*H+m] = h;
      }
    }
  }
}

int RunCase(int64_t B, int64_t L, int64_t IN, int64_t H, int repeats, bool check) {
  const int64_t Din = 3*IN;
  std::vector<float> x(B*L*IN), wt(Din*H), bias(4*H), out(B*L*H, 0.0f), ref(B*L*H, 0.0f);
  for (size_t i = 0; i < x.size(); ++i) x[i] = ((int)(i % 17) - 8) * 0.03f;
  for (size_t i = 0; i < wt.size(); ++i) wt[i] = ((int)(i % 13) - 6) * 0.02f;
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = ((int)(i % 11) - 5) * 0.01f;
  if (check) Ref(B, L, IN, H, x, wt, bias, ref);

  void *d_x=nullptr,*d_w=nullptr,*d_b=nullptr,*d_o=nullptr;
  CHECK_ACL(aclrtMalloc(&d_x, x.size()*4, ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_w, wt.size()*4, ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_b, bias.size()*4, ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_o, out.size()*4, ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMemcpy(d_x, x.size()*4, x.data(), x.size()*4, ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(d_w, wt.size()*4, wt.data(), wt.size()*4, ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(d_b, bias.size()*4, bias.data(), bias.size()*4, ACL_MEMCPY_HOST_TO_DEVICE));

  aclTensor* xt=MakeTensor({B,L,IN},ACL_FLOAT,d_x), *wtt=MakeTensor({Din,H},ACL_FLOAT,d_w);
  aclTensor* bt=MakeTensor({4*H},ACL_FLOAT,d_b), *ot=MakeTensor({B,L,H},ACL_FLOAT,d_o);
  uint64_t ws=0; aclOpExecutor* ex=nullptr;
  CHECK_ACLNN(aclnnSruScanFusedGetWorkspaceSize(xt,wtt,bt,ot,&ws,&ex));
  void* wsp=nullptr; if(ws>0) CHECK_ACL(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_NORMAL_ONLY));
  aclrtStream st=nullptr; CHECK_ACL(aclrtCreateStream(&st));
  CHECK_ACLNN(aclnnSruScanFused(wsp,ws,ex,st));
  CHECK_ACL(aclrtSynchronizeStream(st));

  if(repeats>0){auto t0=std::chrono::steady_clock::now();for(int i=0;i<repeats;++i){uint64_t w2=0;aclOpExecutor*e2=nullptr;CHECK_ACLNN(aclnnSruScanFusedGetWorkspaceSize(xt,wtt,bt,ot,&w2,&e2));CHECK_ACLNN(aclnnSruScanFused(wsp,w2,e2,st));}CHECK_ACL(aclrtSynchronizeStream(st));double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/repeats;std::cout<<"PERF B="<<B<<" L="<<L<<" IN="<<IN<<" H="<<H<<" avg_ms="<<ms<<std::endl;}

  CHECK_ACL(aclrtMemcpy(out.data(), out.size()*4, d_o, out.size()*4, ACL_MEMCPY_DEVICE_TO_HOST));
  if(check){float md=0;for(size_t i=0;i<out.size();++i)md=std::max(md,std::fabs(out[i]-ref[i]));std::cout<<"CHECK B="<<B<<" L="<<L<<" IN="<<IN<<" H="<<H<<" max_diff="<<md<<std::endl;}

  aclDestroyTensor(xt);aclDestroyTensor(wtt);aclDestroyTensor(bt);aclDestroyTensor(ot);
  if(wsp)aclrtFree(wsp);aclrtDestroyStream(st);aclrtFree(d_x);aclrtFree(d_w);aclrtFree(d_b);aclrtFree(d_o);
  return 0;
}
}

int main(int argc, char** argv) {
  int64_t B=4,L=7,IN=16,H=16;int rp=0;bool ck=true;
  if(argc>=5){B=std::stoll(argv[1]);L=std::stoll(argv[2]);IN=std::stoll(argv[3]);H=std::stoll(argv[4]);rp=argc>=6?std::stoi(argv[5]):0;ck=argc>=7?(std::stoi(argv[6])!=0):false;}
  CHECK_ACL(aclInit(nullptr));CHECK_ACL(aclrtSetDevice(0));
  if(!dlopen("libascend_all_ops.so",RTLD_NOW|RTLD_GLOBAL)){std::cerr<<dlerror()<<std::endl;return 1;}
  if(!dlopen("libcust_opmaster_rt2.0.so",RTLD_NOW|RTLD_GLOBAL)){std::cerr<<dlerror()<<std::endl;return 1;}
  if(!dlopen("libcust_opapi.so",RTLD_NOW|RTLD_GLOBAL)){std::cerr<<dlerror()<<std::endl;return 1;}
  int rc=0;rc|=RunCase(B,L,IN,H,rp,ck);
  if(argc<5){rc|=RunCase(8,33,21,32,0,true);rc|=RunCase(16,64,11,64,0,true);rc|=RunCase(8,50,64,64,0,true);}
  return rc;
}
