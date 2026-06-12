/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEEPONET_HOST_H
#define DEEPONET_HOST_H

#include <cstdint>

namespace optiling {

struct DeepOnetTilingData {
    int32_t batchSize;
    int32_t branchDim;
    int32_t trunkDim;
    int32_t latentDim;
    int32_t querySize;
    int32_t tileSize;
    int32_t coreNum;
    int32_t branchWeightSize;
    int32_t trunkWeightSize;
};

}  // namespace optiling

extern "C" int32_t aclnnDeepOnetForward(
    void* branchInputAddr,
    void* trunkInputAddr,
    void* branchWeightsAddr,
    void* trunkWeightsAddr,
    void* outputAddr,
    int32_t batchSize,
    int32_t branchDim,
    int32_t trunkDim,
    int32_t latentDim,
    int32_t querySize,
    void* workspaceAddr,
    uint64_t workspaceSize,
    void* stream
);

extern "C" uint64_t aclnnDeepOnetForwardGetWorkspaceSize(
    int32_t batchSize,
    int32_t branchDim,
    int32_t trunkDim,
    int32_t latentDim,
    int32_t querySize
);

#endif
