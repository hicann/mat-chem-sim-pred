#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "acl/acl.h"
#include "aclnn_tirex_slstm_cell.h"

#define CHECK_ACL(expr) do { auto _ret = (expr); if (_ret != ACL_SUCCESS) { std::cerr << #expr << " failed, ret=" << _ret << std::endl; return 1; } } while (false)
#define CHECK_ACLNN(expr) do { auto _ret = (expr); if (_ret != 0) { std::cerr << #expr << " failed, ret=" << _ret << std::endl; return 1; } } while (false)

namespace {
aclTensor* MakeTensor(const std::vector<int64_t>& dims, aclDataType dtype, void* data) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
  return aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, dims.data(), dims.size(), data);
}
float LogSigmoid(float x) { if (x > 15.0f) x = 15.0f; return -std::log1p(std::exp(-x)); }
float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
void Ref(int64_t B, int64_t S, int64_t H, int64_t Heads, const std::vector<float>& input, const std::vector<float>& R, const std::vector<float>& bias, const std::vector<float>& init, std::vector<float>& out, std::vector<float>& final) {
  const int64_t Hd = H / Heads;
  final = init;
  for (int64_t b = 0; b < B; ++b) {
    bool initialNAllZero = true;
    for (int64_t h = 0; h < H; ++h) if (init[(2 * B + b) * H + h] != 0.0f) initialNAllZero = false;
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t h = 0; h < H; ++h) {
        const int64_t head = h / Hd;
        const int64_t pos = h - head * Hd;
        float raw[4] = {0, 0, 0, 0};
        for (int64_t g = 0; g < 4; ++g) {
          raw[g] = input[(b * S + s) * (4 * H) + g * H + h] + bias[g * H + h];
          for (int64_t j = 0; j < Hd; ++j) {
            const float prev = (s == 0) ? init[(0 * B + b) * H + head * Hd + j] : out[(b * S + s - 1) * H + head * Hd + j];
            raw[g] += prev * R[(head * Hd + j) * (4 * Hd) + g * Hd + pos];
          }
        }
        float prevC = final[(1 * B + b) * H + h];
        float prevN = final[(2 * B + b) * H + h];
        float prevM = final[(3 * B + b) * H + h];
        float logfplusm = prevM + LogSigmoid(raw[1]);
        float mnew = (s == 0 && initialNAllZero) ? raw[0] : std::max(raw[0], logfplusm);
        float igate = std::min(std::exp(raw[0] - mnew), 1.0f);
        float fgate = std::min(std::exp(logfplusm - mnew), 1.0f);
        float zgate = std::tanh(raw[2]);
        float ogate = Sigmoid(raw[3]);
        float cnew = fgate * prevC + igate * zgate;
        float nnew = fgate * prevN + igate;
        float hnew = ogate * cnew / (nnew == 0.0f ? 1.0e-6f : nnew);
        out[(b * S + s) * H + h] = hnew;
        final[(0 * B + b) * H + h] = hnew;
        final[(1 * B + b) * H + h] = cnew;
        final[(2 * B + b) * H + h] = nnew;
        final[(3 * B + b) * H + h] = mnew;
      }
    }
  }
}
int RunCase(int64_t B, int64_t S, int64_t H, int64_t Heads, int repeats, bool check) {
  std::vector<float> input(B*S*4*H), R(Heads*(H/Heads)*4*(H/Heads)), bias(4*H), init(4*B*H, 0), out(B*S*H, 0), final(4*B*H, 0), refOut(B*S*H, 0), refFinal(4*B*H, 0);
  for (size_t i=0;i<input.size();++i) input[i] = (static_cast<int>(i % 17) - 8) * 0.025f;
  for (size_t i=0;i<R.size();++i) R[i] = (static_cast<int>(i % 13) - 6) * 0.01f;
  for (size_t i=0;i<bias.size();++i) bias[i] = (static_cast<int>(i % 11) - 5) * 0.01f;
  if (check) Ref(B,S,H,Heads,input,R,bias,init,refOut,refFinal);
  void *d_input=nullptr,*d_R=nullptr,*d_bias=nullptr,*d_init=nullptr,*d_out=nullptr,*d_final=nullptr;
  CHECK_ACL(aclrtMalloc(&d_input, input.size()*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_R, R.size()*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_bias, bias.size()*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_init, init.size()*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_out, out.size()*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_final, final.size()*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMemcpy(d_input, input.size()*sizeof(float), input.data(), input.size()*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(d_R, R.size()*sizeof(float), R.data(), R.size()*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(d_bias, bias.size()*sizeof(float), bias.data(), bias.size()*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(d_init, init.size()*sizeof(float), init.data(), init.size()*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  aclTensor* input_t=MakeTensor({B,S,4*H}, ACL_FLOAT, d_input);
  aclTensor* r_t=MakeTensor({Heads,H/Heads,4*(H/Heads)}, ACL_FLOAT, d_R);
  aclTensor* b_t=MakeTensor({4*H}, ACL_FLOAT, d_bias);
  aclTensor* init_t=MakeTensor({4,B,H}, ACL_FLOAT, d_init);
  aclTensor* out_t=MakeTensor({B,S,H}, ACL_FLOAT, d_out);
  aclTensor* final_t=MakeTensor({4,B,H}, ACL_FLOAT, d_final);
  uint64_t workspace_size=0; aclOpExecutor* executor=nullptr;
  CHECK_ACLNN(aclnnTirexSlstmCellGetWorkspaceSize(input_t,r_t,b_t,init_t,out_t,final_t,&workspace_size,&executor));
  void* workspace=nullptr; if (workspace_size>0) CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  aclrtStream stream=nullptr; CHECK_ACL(aclrtCreateStream(&stream));
  CHECK_ACLNN(aclnnTirexSlstmCell(workspace, workspace_size, executor, stream));
  CHECK_ACL(aclrtSynchronizeStream(stream));
  if (repeats > 0) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i=0;i<repeats;++i) {
      uint64_t loop_workspace_size=0;
      aclOpExecutor* loop_executor=nullptr;
      CHECK_ACLNN(aclnnTirexSlstmCellGetWorkspaceSize(input_t,r_t,b_t,init_t,out_t,final_t,&loop_workspace_size,&loop_executor));
      if (loop_workspace_size > workspace_size) {
        if (workspace) aclrtFree(workspace);
        workspace_size = loop_workspace_size;
        CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_NORMAL_ONLY));
      }
      CHECK_ACLNN(aclnnTirexSlstmCell(workspace, loop_workspace_size, loop_executor, stream));
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count()/static_cast<double>(repeats);
    std::cout << "PERF B="<<B<<" S="<<S<<" H="<<H<<" avg_ms="<<ms<<std::endl;
  }
  CHECK_ACL(aclrtMemcpy(out.data(), out.size()*sizeof(float), d_out, out.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
  CHECK_ACL(aclrtMemcpy(final.data(), final.size()*sizeof(float), d_final, final.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
  if (check) {
    float maxDiff=0.0f;
    for (size_t i=0;i<out.size();++i) maxDiff = std::max(maxDiff, std::fabs(out[i]-refOut[i]));
    for (size_t i=0;i<final.size();++i) maxDiff = std::max(maxDiff, std::fabs(final[i]-refFinal[i]));
    std::cout << "CHECK B="<<B<<" S="<<S<<" H="<<H<<" max_diff="<<maxDiff<<std::endl;
    if (!(maxDiff < 1.0e-2f)) return 2;
  }
  aclDestroyTensor(input_t); aclDestroyTensor(r_t); aclDestroyTensor(b_t); aclDestroyTensor(init_t); aclDestroyTensor(out_t); aclDestroyTensor(final_t);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtFree(d_input); aclrtFree(d_R); aclrtFree(d_bias); aclrtFree(d_init); aclrtFree(d_out); aclrtFree(d_final);
  return 0;
}
}
int main(int argc, char** argv) {
  int64_t B = 1, S = 2, H = 4, Heads = 1;
  int repeats = 0;
  bool check = true;
  if (argc >= 6) {
    B = std::stoll(argv[1]);
    S = std::stoll(argv[2]);
    H = std::stoll(argv[3]);
    Heads = std::stoll(argv[4]);
    repeats = std::stoi(argv[5]);
    check = argc >= 7 ? (std::stoi(argv[6]) != 0) : false;
  }
  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(0));
  if (dlopen("libascend_all_ops.so", RTLD_NOW | RTLD_GLOBAL) == nullptr) { std::cerr << dlerror() << std::endl; return 1; }
  if (dlopen("libcust_opmaster_rt2.0.so", RTLD_NOW | RTLD_GLOBAL) == nullptr) { std::cerr << dlerror() << std::endl; return 1; }
  if (dlopen("libcust_opapi.so", RTLD_NOW | RTLD_GLOBAL) == nullptr) { std::cerr << dlerror() << std::endl; return 1; }
  int ret = RunCase(B, S, H, Heads, repeats, check);
  aclrtResetDevice(0); aclFinalize();
  return ret;
}
