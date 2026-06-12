/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PINN_HOST_H
#define PINN_HOST_H

#include <cstdint>

namespace optiling {

struct PinnTilingData {
    int32_t batchSize;
    int32_t inputDim;
    int32_t hiddenDim;
    int32_t outputDim;
    int32_t numLayers;
    int32_t tileSize;
    int32_t coreNum;
    int32_t weightsPerLayer;
    int32_t totalWeightSize;
    int32_t activationType;
};

}  // namespace optiling

extern "C" int32_t aclnnPinnFC(
    void* inputAddr,
    void* weightsAddr,
    void* outputAddr,
    void* gradientAddr,
    int32_t batchSize,
    int32_t inputDim,
    int32_t hiddenDim,
    int32_t outputDim,
    int32_t numLayers,
    int32_t activationType,
    void* workspaceAddr,
    uint64_t workspaceSize,
    void* stream
);

extern "C" uint64_t aclnnPinnFCGetWorkspaceSize(
    int32_t batchSize,
    int32_t inputDim,
    int32_t hiddenDim,
    int32_t outputDim,
    int32_t numLayers
);

#endif
