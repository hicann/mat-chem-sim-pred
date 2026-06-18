/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PID_BASIS_GEMM_ACL_SMOKE_H_
#define PID_BASIS_GEMM_ACL_SMOKE_H_

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "acl/acl.h"

namespace pid_basis_gemm {

using GetWorkspaceSizeFn = uint64_t (*)(int64_t batch, int64_t candidates);
using LaunchFn = int32_t (*)(
    void* dot, void* basis_norm, void* y_energy, void* best_sse, void* best_k, void* best_idx, int64_t batch,
    int64_t candidates, void* workspace, uint64_t workspace_size, void* stream);

inline void* MallocDevice(size_t bytes)
{
    void* ptr = nullptr;
    if (aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || ptr == nullptr) {
        throw std::runtime_error("aclrtMalloc failed");
    }
    return ptr;
}

inline int RunAclSmoke(const char* op_name, GetWorkspaceSizeFn get_workspace_size, LaunchFn launch)
{
    const int64_t batch = 2;
    const int64_t candidates = 4;
    const std::vector<float> dot = {1.0f, 3.0f, 6.0f, 2.0f, 1.0f, 4.0f, 2.0f, 8.0f};
    const std::vector<float> basis_norm = {1.0f, 2.0f, 4.0f, 8.0f};
    const std::vector<float> y_energy = {10.0f, 20.0f};
    const std::vector<float> expected_sse = {1.0f, 12.0f};
    const std::vector<float> expected_k = {1.5f, 2.0f};
    const std::vector<int32_t> expected_idx = {2, 1};

    std::vector<float> best_sse(batch, 0.0f);
    std::vector<float> best_k(batch, 0.0f);
    std::vector<int32_t> best_idx(batch, 0);

    void* d_dot = MallocDevice(dot.size() * sizeof(float));
    void* d_norm = MallocDevice(basis_norm.size() * sizeof(float));
    void* d_energy = MallocDevice(y_energy.size() * sizeof(float));
    void* d_sse = MallocDevice(best_sse.size() * sizeof(float));
    void* d_k = MallocDevice(best_k.size() * sizeof(float));
    void* d_idx = MallocDevice(best_idx.size() * sizeof(int32_t));

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclrtMemcpyAsync(d_dot, dot.size() * sizeof(float), dot.data(), dot.size() * sizeof(float),
                     ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(d_norm, basis_norm.size() * sizeof(float), basis_norm.data(), basis_norm.size() * sizeof(float),
                     ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(d_energy, y_energy.size() * sizeof(float), y_energy.data(), y_energy.size() * sizeof(float),
                     ACL_MEMCPY_HOST_TO_DEVICE, stream);

    const uint64_t workspace_size = get_workspace_size(batch, candidates);
    void* workspace = MallocDevice(workspace_size);
    const int32_t ret =
        launch(d_dot, d_norm, d_energy, d_sse, d_k, d_idx, batch, candidates, workspace, workspace_size, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(best_sse.data(), best_sse.size() * sizeof(float), d_sse, best_sse.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(best_k.data(), best_k.size() * sizeof(float), d_k, best_k.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(best_idx.data(), best_idx.size() * sizeof(int32_t), d_idx, best_idx.size() * sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool passed = ret == ACL_SUCCESS;
    for (int64_t i = 0; i < batch; ++i) {
        passed = passed && std::fabs(best_sse[i] - expected_sse[i]) < 1.0e-5f;
        passed = passed && std::fabs(best_k[i] - expected_k[i]) < 1.0e-5f;
        passed = passed && best_idx[i] == expected_idx[i];
    }

    std::cout << op_name << " smoke best_sse=[" << best_sse[0] << ", " << best_sse[1] << "] best_k=[" << best_k[0]
              << ", " << best_k[1] << "] best_idx=[" << best_idx[0] << ", " << best_idx[1] << "]" << std::endl;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    aclrtFree(workspace);
    aclrtFree(d_idx);
    aclrtFree(d_k);
    aclrtFree(d_sse);
    aclrtFree(d_energy);
    aclrtFree(d_norm);
    aclrtFree(d_dot);
    aclrtDestroyStream(stream);
    return passed ? 0 : 2;
}

}  // namespace pid_basis_gemm

#endif  // PID_BASIS_GEMM_ACL_SMOKE_H_
