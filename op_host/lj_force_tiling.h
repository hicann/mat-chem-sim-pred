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
 * \file lj_force_tiling.h
 * \brief LJForce Tiling 参数定义
 */

#ifndef LJ_FORCE_TILING_H
#define LJ_FORCE_TILING_H

#include <cstdint>

namespace optiling {

struct LJForceTilingData {
    int32_t numAtoms;
    int32_t tileSize;
    int32_t coreNum;
    float epsilon;
    float sigma;
    float cutoff;
    float cutoffSq;
    float sigma6;
    float sigma12;
    float eps4;
    float eps24;
};

}  // namespace optiling

#endif  // LJ_FORCE_TILING_H
