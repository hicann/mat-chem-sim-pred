#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "acl/acl.h"
#include "aclnn_ltc_scan_fused.h"

#define CHECK_ACL(expr) do { auto _ret = (expr); if (_ret != ACL_SUCCESS) { std::cerr << #expr << " failed, ret=" << _ret << std::endl; return 1; } } while (false)
#define CHECK_ACLNN(expr) do { auto _ret = (expr); if (_ret != 0) { std::cerr << #expr << " failed, ret=" << _ret << std::endl; return 1; } } while (false)

namespace {

aclTensor* MakeTensor(const std::vector<int64_t>& dims, aclDataType dtype, void* data) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i)
    strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
  return aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, dims.data(), dims.size(), data);
}

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

constexpr float DT = 0.042f;
constexpr int K_UNFOLD = 6;

// Faithful CPU oracle of one LTC cell (Hasani et al. AAAI'21, fused ODE solver), per
// timestep with K inner unfolds:
//   wx = W_in x_s
//   repeat K: pre = W_rec h + wx + b; f = sigmoid(pre);
//             h = (cm/dt*h + gleak*Eleak + f*A) / (cm/dt + gleak + f)
// x   : [B, L, IN]
// wt  : [H+IN, H]  column-packed transposed weight: wt[0:H,:]=W_rec, wt[H:H+IN,:]=W_in
// bias: [5H] packed per-neuron: [ b | gleak | Eleak | cm | A ]
// out : [B, L, H]  (= h)
void Ref(int64_t B, int64_t L, int64_t IN, int64_t H,
         const std::vector<float>& x, const std::vector<float>& wt,
         const std::vector<float>& bias, std::vector<float>& out) {
  const int64_t Din = H + IN;
  const float* bb = bias.data();
  const float* gleak = bias.data() + H;
  const float* Eleak = bias.data() + 2 * H;
  const float* cm = bias.data() + 3 * H;
  const float* Acond = bias.data() + 4 * H;
  std::vector<float> h(H), wx(H);
  for (int64_t b = 0; b < B; ++b) {
    std::fill(h.begin(), h.end(), 0.0f);
    for (int64_t s = 0; s < L; ++s) {
      for (int64_t m = 0; m < H; ++m) {
        float acc = 0.0f;
        for (int64_t k = 0; k < IN; ++k) acc += x[((b * L) + s) * IN + k] * wt[(H + k) * H + m];
        wx[m] = acc;
      }
      for (int u = 0; u < K_UNFOLD; ++u) {
        for (int64_t m = 0; m < H; ++m) {
          float acc = 0.0f;
          for (int64_t c = 0; c < H; ++c) acc += h[c] * wt[c * H + m];
          const float f = Sigmoid(acc + wx[m] + bb[m]);
          const float cmdt = cm[m] / DT;
          h[m] = (cmdt * h[m] + gleak[m] * Eleak[m] + f * Acond[m]) / (cmdt + gleak[m] + f);
        }
      }
      for (int64_t m = 0; m < H; ++m) out[((b * L) + s) * H + m] = h[m];
    }
  }
}

int RunCase(int64_t B, int64_t L, int64_t IN, int64_t H, int repeats, bool check) {
  const int64_t Din = H + IN;
  const int64_t w3 = H;
  const int64_t Nb = 5 * H;
  std::vector<float> x(B * L * IN), wt(Din * w3), bias(Nb), out(B * L * H, 0.0f), refOut(B * L * H, 0.0f);
  for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>(i % 17) - 8) * 0.03f;
  for (size_t i = 0; i < wt.size(); ++i) wt[i] = (static_cast<int>(i % 13) - 6) * 0.02f;
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = (static_cast<int>(i % 11) - 5) * 0.01f;
  // gleak (rows[H:2H]) and cm (rows[3H:4H]) must be positive for a stable denominator
  for (int64_t m = 0; m < H; ++m) { bias[H + m] = 0.5f + (m % 7) * 0.05f; bias[3 * H + m] = 0.5f + (m % 5) * 0.06f; }
  if (check) Ref(B, L, IN, H, x, wt, bias, refOut);

  void *d_x=nullptr,*d_w=nullptr,*d_b=nullptr,*d_out=nullptr;
  CHECK_ACL(aclrtMalloc(&d_x, x.size()*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_w, wt.size()*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_b, bias.size()*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_out, out.size()*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMemcpy(d_x, x.size()*sizeof(float), x.data(), x.size()*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(d_w, wt.size()*sizeof(float), wt.data(), wt.size()*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(d_b, bias.size()*sizeof(float), bias.data(), bias.size()*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

  aclTensor* x_t = MakeTensor({B, L, IN}, ACL_FLOAT, d_x);
  aclTensor* w_t = MakeTensor({Din, w3}, ACL_FLOAT, d_w);
  aclTensor* b_t = MakeTensor({Nb}, ACL_FLOAT, d_b);
  aclTensor* out_t = MakeTensor({B, L, H}, ACL_FLOAT, d_out);

  uint64_t workspace_size = 0; aclOpExecutor* executor = nullptr;
  CHECK_ACLNN(aclnnLtcScanFusedGetWorkspaceSize(x_t, w_t, b_t, out_t, &workspace_size, &executor));
  void* workspace = nullptr; if (workspace_size > 0) CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  aclrtStream stream = nullptr; CHECK_ACL(aclrtCreateStream(&stream));
  CHECK_ACLNN(aclnnLtcScanFused(workspace, workspace_size, executor, stream));
  CHECK_ACL(aclrtSynchronizeStream(stream));

  if (repeats > 0) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
      uint64_t ws = 0; aclOpExecutor* ex = nullptr;
      CHECK_ACLNN(aclnnLtcScanFusedGetWorkspaceSize(x_t, w_t, b_t, out_t, &ws, &ex));
      if (ws > workspace_size) { if (workspace) aclrtFree(workspace); workspace_size = ws; CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_NORMAL_ONLY)); }
      CHECK_ACLNN(aclnnLtcScanFused(workspace, ws, ex, stream));
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(repeats);
    std::cout << "PERF B="<<B<<" L="<<L<<" IN="<<IN<<" H="<<H<<" avg_ms="<<ms<<std::endl;
  }

  CHECK_ACL(aclrtMemcpy(out.data(), out.size()*sizeof(float), d_out, out.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
  if (check) {
    float maxDiff = 0.0f;
    for (size_t i = 0; i < out.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(out[i] - refOut[i]));
    std::cout << "CHECK B="<<B<<" L="<<L<<" IN="<<IN<<" H="<<H<<" max_diff="<<maxDiff<<std::endl;
    if (!(maxDiff < 1.0e-3f)) return 2;
  }

  aclDestroyTensor(x_t); aclDestroyTensor(w_t); aclDestroyTensor(b_t); aclDestroyTensor(out_t);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtFree(d_x); aclrtFree(d_w); aclrtFree(d_b); aclrtFree(d_out);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int64_t B = 4, L = 7, IN = 16, H = 16;
  int repeats = 0;
  bool check = true;
  if (argc >= 5) {
    B = std::stoll(argv[1]); L = std::stoll(argv[2]); IN = std::stoll(argv[3]); H = std::stoll(argv[4]);
    repeats = argc >= 6 ? std::stoi(argv[5]) : 0;
    check = argc >= 7 ? (std::stoi(argv[6]) != 0) : false;
  }
  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(0));
  if (dlopen("libascend_all_ops.so", RTLD_NOW | RTLD_GLOBAL) == nullptr) { std::cerr << dlerror() << std::endl; return 1; }
  if (dlopen("libcust_opmaster_rt2.0.so", RTLD_NOW | RTLD_GLOBAL) == nullptr) { std::cerr << dlerror() << std::endl; return 1; }
  if (dlopen("libcust_opapi.so", RTLD_NOW | RTLD_GLOBAL) == nullptr) { std::cerr << dlerror() << std::endl; return 1; }

  int rc = 0;
  rc |= RunCase(B, L, IN, H, repeats, check);
  if (argc < 5) {
    rc |= RunCase(8, 33, 21, 32, 0, true);
    rc |= RunCase(16, 64, 11, 64, 0, true);
    rc |= RunCase(8, 50, 64, 64, 0, true);
  }
  return rc;
}
