/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deeponet_kernel.h"

extern "C" __global__ __aicore__ void deeponet_forward_kernel(
    GM_ADDR branchInput,
    GM_ADDR trunkInput,
    GM_ADDR branchWeights,
    GM_ADDR trunkWeights,
    GM_ADDR output,
    GM_ADDR tiling
) {
    DeepOnetOp op;
    op.Init(branchInput, trunkInput, branchWeights, trunkWeights, output, tiling);
    op.Process();
}
