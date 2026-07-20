#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <numeric>
#include <vector>

#include "acl/acl.h"
#include "aclnn_auto_corr_fused_aggregate.h"

namespace {

#define CHECK_ACL(expr)                                                       \
  do {                                                                        \
    aclError _ret = (expr);                                                   \
    if (_ret != ACL_SUCCESS) {                                                \
      std::cerr << #expr << " failed, ret=" << _ret << std::endl;             \
      return 1;                                                               \
    }                                                                         \
  } while (false)

aclTensor* MakeTensor(const std::vector<int64_t>& dims, aclDataType dtype, void* data) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
  }
  return aclCreateTensor(
      dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
      dims.data(), dims.size(), data);
}

double NowMs() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

double Mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

float FastExpRef(float x) {
  float y = 1.0f + x / 32.0f;
  for (int i = 0; i < 5; ++i) {
    y *= y;
  }
  return y;
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_abs_diff;
}

void FillInput(std::vector<float>& query, std::vector<float>& key, std::vector<float>& value) {
  for (size_t i = 0; i < query.size(); ++i) {
    const int32_t q = static_cast<int32_t>((i * 37U + 11U) % 101U) - 50;
    query[i] = static_cast<float>(q) / 50.0f;
    key[i] = query[i];
    const int32_t v = static_cast<int32_t>((i * 17U + 5U) % 97U) - 48;
    value[i] = static_cast<float>(v) / 23.0f;
  }
}

void CpuFusedTopK(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    std::vector<float>& output,
    int64_t batch,
    int64_t heads,
    int64_t embed,
    int64_t length,
    int64_t top_k) {
  const int64_t groups = batch * heads * embed;
  for (int64_t group = 0; group < groups; ++group) {
    const int64_t base = group * length;
    std::vector<float> scores(static_cast<size_t>(top_k), -1.0e30f);
    std::vector<int32_t> lags(static_cast<size_t>(top_k), -1);
    for (int64_t lag = 0; lag < length; ++lag) {
      float score = 0.0f;
      const int64_t no_wrap = length - lag;
      for (int64_t t = 0; t < no_wrap; ++t) {
        score += query[static_cast<size_t>(base + t)] *
                 key[static_cast<size_t>(base + t + lag)];
      }
      for (int64_t t = no_wrap; t < length; ++t) {
        score += query[static_cast<size_t>(base + t)] *
                 key[static_cast<size_t>(base + t - no_wrap)];
      }
      if (score <= scores[static_cast<size_t>(top_k - 1)]) {
        continue;
      }
      int64_t pos = top_k - 1;
      while (pos > 0 && score > scores[static_cast<size_t>(pos - 1)]) {
        scores[static_cast<size_t>(pos)] = scores[static_cast<size_t>(pos - 1)];
        lags[static_cast<size_t>(pos)] = lags[static_cast<size_t>(pos - 1)];
        --pos;
      }
      scores[static_cast<size_t>(pos)] = score;
      lags[static_cast<size_t>(pos)] = static_cast<int32_t>(lag);
    }

    float max_value = scores[0];
    for (int64_t k = 1; k < top_k; ++k) {
      max_value = std::max(max_value, scores[static_cast<size_t>(k)]);
    }
    float sum = 0.0f;
    for (int64_t k = 0; k < top_k; ++k) {
      const float e = FastExpRef(scores[static_cast<size_t>(k)] - max_value);
      scores[static_cast<size_t>(k)] = e;
      sum += e;
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int64_t k = 0; k < top_k; ++k) {
      scores[static_cast<size_t>(k)] *= inv;
    }

    for (int64_t t = 0; t < length; ++t) {
      float acc = 0.0f;
      for (int64_t k = 0; k < top_k; ++k) {
        int64_t src_t = t + lags[static_cast<size_t>(k)];
        if (src_t >= length) {
          src_t -= length;
        }
        acc += scores[static_cast<size_t>(k)] * value[static_cast<size_t>(base + src_t)];
      }
      output[static_cast<size_t>(base + t)] = acc;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  int64_t batch = argc > 1 ? std::stoll(argv[1]) : 8;
  int64_t heads = argc > 2 ? std::stoll(argv[2]) : 4;
  int64_t embed = argc > 3 ? std::stoll(argv[3]) : 32;
  int64_t length = argc > 4 ? std::stoll(argv[4]) : 96;
  int64_t top_k = argc > 5 ? std::stoll(argv[5]) : 1;
  int warmup = argc > 6 ? std::stoi(argv[6]) : 3;
  int repeat = argc > 7 ? std::stoi(argv[7]) : 10;
  const int64_t total = batch * heads * embed * length;
  std::vector<float> h_query(static_cast<size_t>(total));
  std::vector<float> h_key(static_cast<size_t>(total));
  std::vector<float> h_value(static_cast<size_t>(total));
  std::vector<float> h_output(static_cast<size_t>(total), 0.0f);
  std::vector<float> h_expected(static_cast<size_t>(total), 0.0f);
  FillInput(h_query, h_key, h_value);

  const double cpu_start = NowMs();
  CpuFusedTopK(h_query, h_key, h_value, h_expected, batch, heads, embed, length, top_k);
  const double cpu_ms = NowMs() - cpu_start;

  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(0));
  void* all_ops = dlopen("libascend_all_ops.so", RTLD_NOW | RTLD_GLOBAL);
  if (all_ops == nullptr) {
    std::cerr << "dlopen libascend_all_ops.so failed: " << dlerror() << std::endl;
    return 1;
  }
  void* opapi = dlopen("libcust_opapi.so", RTLD_NOW | RTLD_GLOBAL);
  if (opapi == nullptr) {
    std::cerr << "dlopen libcust_opapi.so failed: " << dlerror() << std::endl;
    return 1;
  }
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  void *d_query = nullptr, *d_key = nullptr, *d_value = nullptr, *d_output = nullptr;
  CHECK_ACL(aclrtMalloc(&d_query, h_query.size() * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_key, h_key.size() * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_value, h_value.size() * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_output, h_output.size() * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMemcpy(d_query, h_query.size() * sizeof(float), h_query.data(),
                        h_query.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(d_key, h_key.size() * sizeof(float), h_key.data(),
                        h_key.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(d_value, h_value.size() * sizeof(float), h_value.data(),
                        h_value.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

  aclTensor* query = MakeTensor({batch, heads, embed, length}, ACL_FLOAT, d_query);
  aclTensor* key = MakeTensor({batch, heads, embed, length}, ACL_FLOAT, d_key);
  aclTensor* value = MakeTensor({batch, heads, embed, length}, ACL_FLOAT, d_value);
  aclTensor* output = MakeTensor({batch, heads, embed, length}, ACL_FLOAT, d_output);

  auto run_once = [&]() -> double {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const double start = NowMs();
    aclnnStatus nn_ret = aclnnAutoCorrFusedAggregateGetWorkspaceSize(
        query, key, value, top_k, output, &workspace_size, &executor);
    if (nn_ret != 0) {
      std::cerr << "aclnnAutoCorrFusedAggregateGetWorkspaceSize failed, ret=" << nn_ret << std::endl;
      return -1.0;
    }
    void* workspace = nullptr;
    if (workspace_size > 0) {
      aclError acl_ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_NORMAL_ONLY);
      if (acl_ret != ACL_SUCCESS) {
        std::cerr << "aclrtMalloc workspace failed, ret=" << acl_ret << std::endl;
        return -1.0;
      }
    }
    nn_ret = aclnnAutoCorrFusedAggregate(workspace, workspace_size, executor, stream);
    if (nn_ret != 0) {
      std::cerr << "aclnnAutoCorrFusedAggregate failed, ret=" << nn_ret << std::endl;
      if (workspace != nullptr) {
        aclrtFree(workspace);
      }
      return -1.0;
    }
    aclError acl_ret = aclrtSynchronizeStream(stream);
    if (acl_ret != ACL_SUCCESS) {
      std::cerr << "aclrtSynchronizeStream failed, ret=" << acl_ret << std::endl;
      if (workspace != nullptr) {
        aclrtFree(workspace);
      }
      return -1.0;
    }
    const double elapsed = NowMs() - start;
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    return elapsed;
  };

  const double cold_ms = run_once();
  if (cold_ms < 0.0) {
    return 1;
  }
  for (int i = 0; i < warmup; ++i) {
    if (run_once() < 0.0) {
      return 1;
    }
  }
  std::vector<double> hot_ms;
  hot_ms.reserve(static_cast<size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    const double elapsed = run_once();
    if (elapsed < 0.0) {
      return 1;
    }
    hot_ms.push_back(elapsed);
  }

  CHECK_ACL(aclrtMemcpy(h_output.data(), h_output.size() * sizeof(float), d_output,
                        h_output.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
  const float max_abs_diff = MaxAbsDiff(h_output, h_expected);
  const double npu_hot = Mean(hot_ms);
  std::cout << "name,shape,cpu_ref_ms,npu_cold_ms,npu_hot_ms,max_abs_diff,npu_hot_vs_cpu\n";
  std::cout << "AutoCorrFusedAggregate,"
            << batch << "x" << heads << "x" << embed << "x" << length << "xK" << top_k << ","
            << cpu_ms << "," << cold_ms << "," << npu_hot << "," << max_abs_diff << ","
            << (cpu_ms / npu_hot) << "\n";

  aclDestroyTensor(query);
  aclDestroyTensor(key);
  aclDestroyTensor(value);
  aclDestroyTensor(output);
  aclrtFree(d_query);
  aclrtFree(d_key);
  aclrtFree(d_value);
  aclrtFree(d_output);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return max_abs_diff <= 1e-4f ? 0 : 2;
}
