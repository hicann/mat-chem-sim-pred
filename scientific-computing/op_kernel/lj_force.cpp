/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lj_force.cpp
 * \brief LJ Force Kernel Entry Point
 */

#include "lj_force.h"

extern "C" __global__ __aicore__ void lj_force_kernel(
    GM_ADDR positions,
    GM_ADDR forces,
    GM_ADDR energy,
    GM_ADDR tiling
) {
    LJForceOp op;
    op.Init(positions, forces, energy, tiling);
    op.Process();
}
