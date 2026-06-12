/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mesh_graph_net_kernel.h"

extern "C" __global__ __aicore__ void mesh_graph_net_kernel(
    GM_ADDR nodeFeatures,
    GM_ADDR edgeIndices,
    GM_ADDR edgeFeatures,
    GM_ADDR nodeWeights,
    GM_ADDR edgeWeights,
    GM_ADDR output,
    GM_ADDR tiling
) {
    MeshGraphNetOp op;
    op.Init(nodeFeatures, edgeIndices, edgeFeatures,
             nodeWeights, edgeWeights, output, tiling);
    op.Process();
}
