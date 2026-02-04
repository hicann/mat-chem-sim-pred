/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @author Liu Fei (@Magic_LF)
 */

/*!
 * \file lj_force.cpp
 * \brief LJForce Host 端实现 (Tiling 计算与 Kernel 启动)
 */

#include "acl/acl.h"
#include "lj_force_tiling.h"
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace optiling;

inline uint32_t AlignUp32(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

extern "C" void aclrtlaunch_lj_force_kernel(uint32_t blockDim, aclrtStream stream,
                                             void* positions, void* forces, void* energy,
                                             void* tiling);

void ComputeLJTiling(int32_t numAtoms, float epsilon, float sigma, float cutoff,
                     LJForceTilingData& tiling, int32_t maxCores = 32) {
    int32_t minAtomsPerCore = 16;
    int32_t optimalCores = (numAtoms + minAtomsPerCore - 1) / minAtomsPerCore;
    optimalCores = std::min(optimalCores, maxCores);
    optimalCores = std::max(optimalCores, 1);

    int32_t tileSize = (numAtoms + optimalCores - 1) / optimalCores;

    float sigma2 = sigma * sigma;
    float sigma6 = sigma2 * sigma2 * sigma2;
    float sigma12 = sigma6 * sigma6;

    tiling.numAtoms = numAtoms;
    tiling.tileSize = tileSize;
    tiling.coreNum = optimalCores;
    tiling.epsilon = epsilon;
    tiling.sigma = sigma;
    tiling.cutoff = cutoff;
    tiling.cutoffSq = cutoff * cutoff;
    tiling.sigma6 = sigma6;
    tiling.sigma12 = sigma12;
    tiling.eps4 = 4.0f * epsilon;
    tiling.eps24 = 24.0f * epsilon;
}

extern "C" int32_t aclnnLJForceDirect(
    void* positionsAddr,
    void* forcesAddr,
    void* energyAddr,
    int32_t numAtoms,
    float epsilon,
    float sigma,
    float cutoff,
    void* workspaceAddr,
    uint64_t workspaceSize,
    void* stream
) {
    LJForceTilingData tiling;
    ComputeLJTiling(numAtoms, epsilon, sigma, cutoff, tiling);

    aclrtMemcpyAsync(workspaceAddr, sizeof(LJForceTilingData),
                     &tiling, sizeof(LJForceTilingData),
                     ACL_MEMCPY_HOST_TO_DEVICE, (aclrtStream)stream);
    aclrtSynchronizeStream((aclrtStream)stream);

    aclrtlaunch_lj_force_kernel(tiling.coreNum, (aclrtStream)stream,
                                 positionsAddr, forcesAddr, energyAddr,
                                 workspaceAddr);

    return 0;
}

extern "C" uint64_t aclnnLJForceGetWorkspaceSize(int32_t numAtoms) {
    return AlignUp32(sizeof(LJForceTilingData), 32) + 32 * sizeof(float);
}
