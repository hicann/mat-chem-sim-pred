/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PDE_MATH_H
#define PDE_MATH_H

#include "kernel_operator.h"

using namespace AscendC;

namespace pde_math {

constexpr float FNO_PI = 3.14159265358979f;

__aicore__ inline float pde_expf(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 3.402823e+38f;
    float n = floorf(x * 1.442695f + 0.5f);
    float r = x - n * 0.693147f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.166667f + r * (0.041667f + r * (0.008333f + r * 0.001389f)))));
    uint32_t i = (uint32_t)(n + 127.0f);
    uint32_t* pi = (uint32_t*)&p;
    *pi = (*pi & 0x007fffff) | (i << 23);
    return p;
}

__aicore__ inline float pde_tanhf(float x) {
    if (x > 8.0f) return 1.0f;
    if (x < -8.0f) return -1.0f;
    float p = pde_expf(2.0f * x);
    return (p - 1.0f) / (p + 1.0f);
}

__aicore__ inline float pde_sigmoidf(float x) {
    return 1.0f / (1.0f + pde_expf(-x));
}

__aicore__ inline float pde_reluf(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

__aicore__ inline float pde_cosf(float x) {
    while (x > FNO_PI) x -= 2.0f * FNO_PI;
    while (x < -FNO_PI) x += 2.0f * FNO_PI;
    float x2 = x * x;
    return 1.0f - x2 * 0.5f + x2 * x2 * 0.0416667f - x2 * x2 * x2 * 0.00138889f;
}

__aicore__ inline float pde_sinf(float x) {
    while (x > FNO_PI) x -= 2.0f * FNO_PI;
    while (x < -FNO_PI) x += 2.0f * FNO_PI;
    float x2 = x * x;
    return x - x * x2 * 0.166667f + x * x2 * x2 * 0.00833333f - x * x2 * x2 * x2 * 0.000198413f;
}

__aicore__ inline float pde_gelu_f(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + pde_tanhf(0.79788456f * (x + 0.044715f * x3)));
}

}  // namespace pde_math

#endif
