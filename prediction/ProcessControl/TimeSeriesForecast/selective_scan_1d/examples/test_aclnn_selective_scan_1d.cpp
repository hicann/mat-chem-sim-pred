/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "selective_scan_1d_host.h"

namespace {

void CheckAcl(aclError ret, const char* expr)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << expr << " failed, ret=" << ret << std::endl;
        std::exit(1);
    }
}

#define CHECK_ACL(expr) CheckAcl((expr), #expr)

void* MallocDevice(size_t bytes)
{
    void* ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(lhs[i] - rhs[i]));
    }
    return max_abs_diff;
}

}  // namespace

int main(int argc, char** argv)
{
    const int device_id = argc > 1 ? std::atoi(argv[1]) : 0;
    constexpr int64_t batch = 1;
    constexpr int64_t length = 2;
    constexpr int64_t dim = 1;
    constexpr int64_t state = 32;

    const std::vector<float> u{2.0f, 3.0f};
    const std::vector<float> delta{1.0f, 1.0f};
    const std::vector<float> a(static_cast<size_t>(dim * state), 0.0f);
    const std::vector<float> b(static_cast<size_t>(batch * length * state), 1.0f);
    const std::vector<float> c(static_cast<size_t>(batch * length * state), 1.0f);
    const std::vector<float> d{0.0f};
    const std::vector<float> expected{64.0f, 160.0f};
    std::vector<float> output(u.size(), 0.0f);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* d_u = MallocDevice(u.size() * sizeof(float));
    void* d_delta = MallocDevice(delta.size() * sizeof(float));
    void* d_a = MallocDevice(a.size() * sizeof(float));
    void* d_b = MallocDevice(b.size() * sizeof(float));
    void* d_c = MallocDevice(c.size() * sizeof(float));
    void* d_d = MallocDevice(d.size() * sizeof(float));
    void* d_output = MallocDevice(output.size() * sizeof(float));
    const uint64_t workspace_size = aclnnSelectiveScan1DGetWorkspaceSize(batch, length, dim, state);
    void* workspace = MallocDevice(workspace_size);

    CHECK_ACL(aclrtMemcpy(d_u, u.size() * sizeof(float), u.data(), u.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_delta, delta.size() * sizeof(float), delta.data(), delta.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_a, a.size() * sizeof(float), a.data(), a.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_b, b.size() * sizeof(float), b.data(), b.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_c, c.size() * sizeof(float), c.data(), c.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_d, d.size() * sizeof(float), d.data(), d.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(static_cast<aclError>(aclnnSelectiveScan1D(
        d_u, d_delta, d_a, d_b, d_c, d_d, d_output, batch, length, dim, state, workspace, workspace_size, stream)));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(output.data(), output.size() * sizeof(float), d_output, output.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    const float max_abs_diff = MaxAbsDiff(output, expected);
    const bool passed = max_abs_diff <= 1.0e-4f;
    std::cout << "SelectiveScan1D smoke output:";
    for (float value : output) {
        std::cout << " " << value;
    }
    std::cout << " expected: 64 160 max_abs_diff=" << max_abs_diff << std::endl;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    aclrtFree(workspace);
    aclrtFree(d_output);
    aclrtFree(d_d);
    aclrtFree(d_c);
    aclrtFree(d_b);
    aclrtFree(d_a);
    aclrtFree(d_delta);
    aclrtFree(d_u);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return passed ? 0 : 2;
}
