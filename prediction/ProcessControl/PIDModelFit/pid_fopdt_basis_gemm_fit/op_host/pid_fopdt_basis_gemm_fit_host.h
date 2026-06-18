/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PID_FOPDT_BASIS_GEMM_FIT_HOST_H_
#define PID_FOPDT_BASIS_GEMM_FIT_HOST_H_

#include <cstdint>

extern "C" int32_t aclnnPidFopdtBasisGemmFit(
    void* dot, void* basis_norm, void* y_energy, void* best_sse, void* best_k, void* best_idx, int64_t batch,
    int64_t candidates, void* workspace, uint64_t workspace_size, void* stream);

extern "C" uint64_t aclnnPidFopdtBasisGemmFitGetWorkspaceSize(int64_t batch, int64_t candidates);

#endif  // PID_FOPDT_BASIS_GEMM_FIT_HOST_H_
