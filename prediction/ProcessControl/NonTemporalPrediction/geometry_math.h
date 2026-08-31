/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#pragma once

#include "kernel_operator.h"

namespace MolecularGeometry
{
constexpr float kTiny = 1.0e-20F;
constexpr float kPi = 3.14159265358979323846F;
constexpr float kHalfPi = 1.57079632679489661923F;
constexpr float kQuarterPi = 0.78539816339744830962F;
constexpr float kTanPiOverEight = 0.41421356237309504880F;

__aicore__ inline float Dot(const float left[3], const float right[3])
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

__aicore__ inline float CrossSquare(const float left[3], const float right[3])
{
    const float x = left[1] * right[2] - left[2] * right[1];
    const float y = left[2] * right[0] - left[0] * right[2];
    const float z = left[0] * right[1] - left[1] * right[0];
    return x * x + y * y + z * z;
}

__aicore__ inline float ApproximateAtan(float value)
{
    float reduced = value;
    float base = 0.0F;
    if (value > kTanPiOverEight)
    {
        reduced = (value - 1.0F) / (value + 1.0F);
        base = kQuarterPi;
    }
    const float square = reduced * reduced;
    float polynomial = 0.0805374449538F;
    polynomial = polynomial * square - 0.138776856032F;
    polynomial = polynomial * square + 0.199777106478F;
    polynomial = polynomial * square - 0.333329491539F;
    return base + reduced + reduced * square * polynomial;
}

__aicore__ inline float AngleFromValues(float dot, float crossSquare, AscendC::LocalTensor<float>& rootInput,
                                        AscendC::LocalTensor<float>& rootOutput)
{
    AscendC::Duplicate(rootInput, 0.0F, 8U);
    pipe_barrier(PIPE_ALL);
    rootInput.SetValue(0U, crossSquare);
    pipe_barrier(PIPE_ALL);
    AscendC::Sqrt(rootOutput, rootInput, 8U);
    pipe_barrier(PIPE_ALL);
    const float cross = rootOutput.GetValue(0U);
    const float absoluteDot = dot < 0.0F ? -dot : dot;
    if (cross <= kTiny && absoluteDot <= kTiny)
    {
        return 0.0F;
    }
    const float safeCross = cross > kTiny ? cross : kTiny;
    const float safeAbsoluteDot = absoluteDot > kTiny ? absoluteDot : kTiny;
    const float acute = cross <= absoluteDot ? ApproximateAtan(safeCross / safeAbsoluteDot)
                                             : kHalfPi - ApproximateAtan(safeAbsoluteDot / safeCross);
    return dot < 0.0F ? kPi - acute : acute;
}
}  // namespace MolecularGeometry
