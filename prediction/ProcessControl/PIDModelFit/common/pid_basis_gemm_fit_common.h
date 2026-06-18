/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PID_BASIS_GEMM_FIT_COMMON_H_
#define PID_BASIS_GEMM_FIT_COMMON_H_

#include <cstdint>

namespace pid_basis_gemm {

struct PidBasisGemmFitTilingData {
    uint32_t batch;
    uint32_t candidates;
    uint32_t core_num;
    uint32_t reserved;
};

inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return ((value + align - 1U) / align) * align;
}

inline uint32_t ComputeCoreNum(int64_t batch)
{
    if (batch <= 0) {
        return 1U;
    }
    constexpr uint32_t kMaxCoreNum = 24U;
    constexpr uint32_t kLoopsPerTaskUnit = 16U;
    const uint32_t task_units = (static_cast<uint32_t>(batch) + kLoopsPerTaskUnit - 1U) / kLoopsPerTaskUnit;
    return task_units > kMaxCoreNum ? kMaxCoreNum : (task_units == 0U ? 1U : task_units);
}

inline uint64_t GetWorkspaceSize()
{
    return AlignUp(sizeof(PidBasisGemmFitTilingData), 32U);
}

}  // namespace pid_basis_gemm

#endif  // PID_BASIS_GEMM_FIT_COMMON_H_
