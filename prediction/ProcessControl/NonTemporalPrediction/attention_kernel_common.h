/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef ATTENTION_KERNEL_COMMON_H
#define ATTENTION_KERNEL_COMMON_H

#include "attention_fused_tiling.h"
#include "kernel_operator.h"

namespace AttentionKernel
{
struct Shape
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t heads;
    uint32_t channels;
    uint32_t maxSegmentSize;
    uint32_t segmentStride;
    uint32_t outputValues;
    uint32_t outputStride;
};

__aicore__ inline Shape ReadShape(const __gm__ AttentionFusedTiling* tiling)
{
    const uint32_t outputValues = tiling->heads * tiling->channels;
    return {tiling->nodes,          tiling->edges,
            tiling->heads,          tiling->channels,
            tiling->maxSegmentSize, (tiling->maxSegmentSize + 7U) / 8U * 8U,
            outputValues,           (outputValues + 7U) / 8U * 8U};
}

__aicore__ inline uint32_t SegmentSize(AscendC::GlobalTensor<int32_t>& rowPtr,
                                       AscendC::GlobalTensor<int32_t>& sourceIndex, uint32_t target, const Shape& shape,
                                       int32_t* begin)
{
    *begin = rowPtr.GetValue(target);
    const int32_t end = rowPtr.GetValue(target + 1U);
    bool valid = *begin >= 0 && end >= *begin && static_cast<uint32_t>(end) <= shape.edges;
    valid = valid && static_cast<uint32_t>(end - *begin) <= shape.maxSegmentSize;
    for (int32_t edge = *begin; valid && edge < end; ++edge)
    {
        const int32_t source = sourceIndex.GetValue(edge);
        valid = source >= 0 && static_cast<uint32_t>(source) < shape.nodes;
    }
    return valid ? static_cast<uint32_t>(end - *begin) : 0U;
}

__aicore__ inline float Normalize(AscendC::LocalTensor<float>& weights, uint32_t count, float maximum,
                                  AscendC::TEventID scalarToVector, AscendC::TEventID vectorToScalar)
{
    AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);
    AscendC::Adds(weights, weights, -maximum, count);
    AscendC::Exp(weights, weights, count);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);
    float denominator = 0.0F;
    for (uint32_t item = 0U; item < count; ++item)
    {
        denominator += weights.GetValue(item);
    }
    return 1.0F / denominator;
}

__aicore__ inline void Accumulate(AscendC::LocalTensor<float>& output, AscendC::LocalTensor<float>& weights,
                                  AscendC::GlobalTensor<int32_t>& sourceIndex, AscendC::GlobalTensor<float>& values,
                                  int32_t begin, uint32_t count, uint32_t head, const Shape& shape, float inverse)
{
    const uint32_t destination = head * shape.channels;
    for (uint32_t item = 0U; item < count; ++item)
    {
        const float weight = weights.GetValue(item) * inverse;
        const uint32_t source = static_cast<uint32_t>(sourceIndex.GetValue(begin + item));
        const uint64_t sourceBase = (static_cast<uint64_t>(source) * shape.heads + head) * shape.channels;
        for (uint32_t channel = 0U; channel < shape.channels; ++channel)
        {
            const uint32_t offset = destination + channel;
            output.SetValue(offset, output.GetValue(offset) + weight * values.GetValue(sourceBase + channel));
        }
    }
}
}  // namespace AttentionKernel

#endif  // ATTENTION_KERNEL_COMMON_H
