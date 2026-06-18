/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pid_basis_gemm_fit_kernel.h"

extern "C" __global__ __aicore__ void pid_fopdt_basis_gemm_fit_kernel(
    GM_ADDR dot, GM_ADDR basis_norm, GM_ADDR y_energy, GM_ADDR best_sse, GM_ADDR best_k, GM_ADDR best_idx,
    GM_ADDR tiling)
{
    pid_basis_gemm::PidBasisGemmFitReduceOp op;
    op.Init(dot, basis_norm, y_energy, best_sse, best_k, best_idx, tiling);
    op.Process();
}
