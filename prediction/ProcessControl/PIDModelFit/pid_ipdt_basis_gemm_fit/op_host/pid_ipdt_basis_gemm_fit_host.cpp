/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pid_ipdt_basis_gemm_fit_host.h"

#include "acl/acl.h"
#include "pid_basis_gemm_fit_host_impl.h"

extern "C" void aclrtlaunch_pid_ipdt_basis_gemm_fit_kernel(
    uint32_t blockDim, aclrtStream stream, void* dot, void* basis_norm, void* y_energy, void* best_sse, void* best_k,
    void* best_idx, void* tiling);

extern "C" int32_t aclnnPidIpdtBasisGemmFit(
    void* dot, void* basis_norm, void* y_energy, void* best_sse, void* best_k, void* best_idx, int64_t batch,
    int64_t candidates, void* workspace, uint64_t workspace_size, void* stream)
{
    return pid_basis_gemm::LaunchBasisGemmFit(
        dot, basis_norm, y_energy, best_sse, best_k, best_idx, batch, candidates, workspace, workspace_size, stream,
        aclrtlaunch_pid_ipdt_basis_gemm_fit_kernel);
}

extern "C" uint64_t aclnnPidIpdtBasisGemmFitGetWorkspaceSize(int64_t batch, int64_t candidates)
{
    (void)batch;
    (void)candidates;
    return pid_basis_gemm::GetWorkspaceSize();
}
