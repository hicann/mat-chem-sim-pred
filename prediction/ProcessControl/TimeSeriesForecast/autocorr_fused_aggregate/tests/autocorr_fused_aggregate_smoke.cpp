#include <algorithm>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_auto_corr_fused_aggregate.h"

namespace {

#define CHECK_ACL(expr)                                                     \
  do {                                                                      \
    aclError _ret = (expr);                                                 \
    if (_ret != ACL_SUCCESS) {                                              \
      std::cerr << #expr << " failed, ret=" << _ret << std::endl;           \
      return 1;                                                             \
    }                                                                       \
  } while (false)

#define CHECK_ACLNN(expr)                                                   \
  do {                                                                      \
    aclnnStatus _ret = (expr);                                              \
    if (_ret != 0) {                                                        \
      std::cerr << #expr << " failed, ret=" << _ret << std::endl;           \
      return 1;                                                             \
    }                                                                       \
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

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_abs_diff;
}

}  // namespace

int main() {
  constexpr int32_t kDeviceId = 0;
  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(kDeviceId));

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

  const std::vector<float> h_query{1.0f, 0.0f, 0.0f, 0.0f};
  const std::vector<float> h_key{1.0f, 0.0f, 0.0f, 0.0f};
  const std::vector<float> h_value{10.0f, 20.0f, 30.0f, 40.0f};
  const std::vector<float> h_expected{10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> h_output(4, 0.0f);

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

  aclTensor* query = MakeTensor({1, 1, 1, 4}, ACL_FLOAT, d_query);
  aclTensor* key = MakeTensor({1, 1, 1, 4}, ACL_FLOAT, d_key);
  aclTensor* value = MakeTensor({1, 1, 1, 4}, ACL_FLOAT, d_value);
  aclTensor* output = MakeTensor({1, 1, 1, 4}, ACL_FLOAT, d_output);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACLNN(aclnnAutoCorrFusedAggregateGetWorkspaceSize(
      query, key, value, 1, output, &workspace_size, &executor));
  void* workspace = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  }
  CHECK_ACLNN(aclnnAutoCorrFusedAggregate(workspace, workspace_size, executor, stream));
  CHECK_ACL(aclrtSynchronizeStream(stream));
  CHECK_ACL(aclrtMemcpy(h_output.data(), h_output.size() * sizeof(float), d_output,
                        h_output.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

  const float max_abs_diff = MaxAbsDiff(h_output, h_expected);
  std::cout << "output:";
  for (float v : h_output) {
    std::cout << " " << v;
  }
  std::cout << "\nexpected: 10 20 30 40\n";
  std::cout << "max_abs_diff: " << max_abs_diff << "\n";

  aclDestroyTensor(query);
  aclDestroyTensor(key);
  aclDestroyTensor(value);
  aclDestroyTensor(output);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  aclrtFree(d_query);
  aclrtFree(d_key);
  aclrtFree(d_value);
  aclrtFree(d_output);
  aclrtDestroyStream(stream);
  aclrtResetDevice(kDeviceId);
  aclFinalize();
  return max_abs_diff <= 1e-4f ? 0 : 2;
}
