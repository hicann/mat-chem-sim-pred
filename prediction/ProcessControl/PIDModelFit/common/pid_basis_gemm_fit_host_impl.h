/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PID_BASIS_GEMM_FIT_HOST_IMPL_H_
#define PID_BASIS_GEMM_FIT_HOST_IMPL_H_

#include "acl/acl.h"
#include "pid_basis_gemm_fit_common.h"

namespace pid_basis_gemm {

using KernelLauncher = void (*)(
    uint32_t blockDim, aclrtStream stream, void* dot, void* basis_norm, void* y_energy, void* best_sse, void* best_k,
    void* best_idx, void* tiling);

inline int32_t LaunchBasisGemmFit(
    void* dot, void* basis_norm, void* y_energy, void* best_sse, void* best_k, void* best_idx, int64_t batch,
    int64_t candidates, void* workspace, uint64_t workspace_size, void* stream, KernelLauncher launcher)
{
    if (dot == nullptr || basis_norm == nullptr || y_energy == nullptr || best_sse == nullptr || best_k == nullptr ||
        best_idx == nullptr || workspace == nullptr || stream == nullptr || launcher == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (batch <= 0 || candidates <= 0 || workspace_size < GetWorkspaceSize()) {
        return ACL_ERROR_INVALID_PARAM;
    }

    PidBasisGemmFitTilingData tiling;
    tiling.batch = static_cast<uint32_t>(batch);
    tiling.candidates = static_cast<uint32_t>(candidates);
    tiling.core_num = ComputeCoreNum(batch);
    tiling.reserved = 0U;

    const auto ret = aclrtMemcpyAsync(
        workspace, sizeof(PidBasisGemmFitTilingData), &tiling, sizeof(PidBasisGemmFitTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE, reinterpret_cast<aclrtStream>(stream));
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    launcher(
        tiling.core_num, reinterpret_cast<aclrtStream>(stream), dot, basis_norm, y_energy, best_sse, best_k, best_idx,
        workspace);
    return ACL_SUCCESS;
}

}  // namespace pid_basis_gemm

#endif  // PID_BASIS_GEMM_FIT_HOST_IMPL_H_
