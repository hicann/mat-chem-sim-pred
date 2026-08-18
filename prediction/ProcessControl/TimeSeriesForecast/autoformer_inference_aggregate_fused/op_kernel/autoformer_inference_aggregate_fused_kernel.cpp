/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace
{

struct AutocorrTopkTiling
{
    uint32_t batch;
    uint32_t heads;
    uint32_t channels;
    uint32_t length;
    uint32_t top_k;
    float inverse_heads;
    float inverse_channels;
};

class AutocorrInferenceFusedKernel
{
   public:
    __aicore__ inline AutocorrInferenceFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR values, GM_ADDR correlation, GM_ADDR output, const AutocorrTopkTiling& tiling)
    {
        batch_ = tiling.batch;
        heads_ = tiling.heads;
        channels_ = tiling.channels;
        length_ = tiling.length;
        topK_ = tiling.top_k;
        inverseHeads_ = tiling.inverse_heads;
        inverseChannels_ = tiling.inverse_channels;
        const uint64_t elements = static_cast<uint64_t>(batch_) * heads_ * channels_ * length_;
        valuesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values), elements);
        correlationGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(correlation), elements);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), elements);
        const uint32_t alignedLength = ((length_ + 7U) / 8U) * 8U;
        pipe_.InitBuffer(meanBuffer_, alignedLength * sizeof(float));
        pipe_.InitBuffer(channelBuffer_, alignedLength * sizeof(float));
        pipe_.InitBuffer(rowBuffer_, alignedLength * sizeof(float));
        pipe_.InitBuffer(topValueBuffer_, 16U * sizeof(float));
        pipe_.InitBuffer(topIndexBuffer_, 16U * sizeof(int64_t));
        pipe_.InitBuffer(reduceWorkBuffer_, 32U * sizeof(float));
        pipe_.InitBuffer(reduceSumBuffer_, 32U * sizeof(float));
        pipe_.InitBuffer(inputBuffer_, alignedLength * sizeof(float));
        pipe_.InitBuffer(shiftedBuffer_, alignedLength * sizeof(float));
        pipe_.InitBuffer(accumulatorBuffer_, alignedLength * sizeof(float));
        pipe_.InitBuffer(offsetBuffer_, alignedLength * sizeof(uint32_t));
    }

    __aicore__ inline void ComputeMean(uint64_t batch, uint64_t batchStride)
    {
        LocalTensor<float> mean = meanBuffer_.Get<float>();
        LocalTensor<float> channelMean = channelBuffer_.Get<float>();
        LocalTensor<float> row = rowBuffer_.Get<float>();
        Duplicate(mean, 0.0F, length_);
        for (uint32_t channel = 0; channel < channels_; ++channel)
        {
            Duplicate(channelMean, 0.0F, length_);
            for (uint32_t head = 0; head < heads_; ++head)
            {
                const uint64_t source = static_cast<uint64_t>(batch) * batchStride +
                                        (static_cast<uint64_t>(head) * channels_ + channel) * length_;
                DataCopy(row, correlationGm_[source], length_);
                pipe_barrier(PIPE_ALL);
                Add(channelMean, channelMean, row, length_);
                pipe_barrier(PIPE_V);
            }
            Muls(channelMean, channelMean, inverseHeads_, length_);
            Add(mean, mean, channelMean, length_);
            pipe_barrier(PIPE_V);
        }
        Muls(mean, mean, inverseChannels_, length_);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void SelectTopDelays()
    {
        LocalTensor<float> mean = meanBuffer_.Get<float>();
        LocalTensor<float> topValues = topValueBuffer_.Get<float>();
        LocalTensor<int64_t> topIndices = topIndexBuffer_.Get<int64_t>();
        for (uint32_t rank = 0; rank < 16U; ++rank)
        {
            topValues.SetValue(rank, -3.4e38F);
            topIndices.SetValue(rank, -1);
        }
        for (uint32_t position = 0; position < length_; ++position)
        {
            const float value = mean.GetValue(position);
            for (uint32_t rank = 0; rank < topK_; ++rank)
            {
                if (value > topValues.GetValue(rank))
                {
                    for (uint32_t tail = topK_ - 1U; tail > rank; --tail)
                    {
                        topValues.SetValue(tail, topValues.GetValue(tail - 1U));
                        topIndices.SetValue(tail, topIndices.GetValue(tail - 1U));
                    }
                    topValues.SetValue(rank, value);
                    topIndices.SetValue(rank, static_cast<int64_t>(position));
                    break;
                }
            }
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void NormalizeTopWeights()
    {
        LocalTensor<float> topValues = topValueBuffer_.Get<float>();
        LocalTensor<float> reduceWork = reduceWorkBuffer_.Get<float>();
        LocalTensor<float> reduceSum = reduceSumBuffer_.Get<float>();
        const float maximum = topValues.GetValue(0);
        Adds(topValues, topValues, -maximum, topK_);
        Exp(topValues, topValues, topK_);
        ReduceSum(reduceSum, topValues, reduceWork, topK_);
        pipe_barrier(PIPE_ALL);
        const float inverseSum = 1.0F / reduceSum.GetValue(0);
        Muls(topValues, topValues, inverseSum, topK_);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void BuildOffsets(int64_t delay)
    {
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<uint32_t>();
        for (uint32_t position = 0; position < length_; ++position)
        {
            const uint32_t source = static_cast<uint32_t>((static_cast<int64_t>(position) + delay) % length_);
            offsets.SetValue(position, source * sizeof(float));
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void GatherShifted(LocalTensor<float> input, LocalTensor<float> shifted)
    {
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<uint32_t>();
        constexpr uint32_t kChunk = 128U;
        for (uint32_t start = 0; start < length_; start += kChunk)
        {
            const uint32_t count = length_ - start < kChunk ? length_ - start : kChunk;
            Gather(shifted[start], input, offsets[start], 0U, count);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void AggregateRow(uint64_t rowBase)
    {
        LocalTensor<float> topValues = topValueBuffer_.Get<float>();
        LocalTensor<int64_t> topIndices = topIndexBuffer_.Get<int64_t>();
        LocalTensor<float> input = inputBuffer_.Get<float>();
        LocalTensor<float> shifted = shiftedBuffer_.Get<float>();
        LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        DataCopy(input, valuesGm_[rowBase], length_);
        Duplicate(accumulator, 0.0F, length_);
        pipe_barrier(PIPE_ALL);
        for (uint32_t rank = 0; rank < topK_; ++rank)
        {
            BuildOffsets(topIndices.GetValue(rank));
            GatherShifted(input, shifted);
            Muls(shifted, shifted, topValues.GetValue(rank), length_);
            pipe_barrier(PIPE_V);
            Add(accumulator, accumulator, shifted, length_);
            pipe_barrier(PIPE_V);
        }
        DataCopy(outputGm_[rowBase], accumulator, length_);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void Process()
    {
        const uint64_t rowsPerBatch = static_cast<uint64_t>(heads_) * channels_;
        const uint64_t batchStride = rowsPerBatch * length_;
        for (uint64_t batch = GetBlockIdx(); batch < batch_; batch += GetBlockNum())
        {
            ComputeMean(batch, batchStride);
            SelectTopDelays();
            NormalizeTopWeights();
            for (uint64_t row = 0; row < rowsPerBatch; ++row)
            {
                AggregateRow(batch * batchStride + row * length_);
            }
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> meanBuffer_;
    TBuf<TPosition::VECCALC> channelBuffer_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    TBuf<TPosition::VECCALC> topValueBuffer_;
    TBuf<TPosition::VECCALC> topIndexBuffer_;
    TBuf<TPosition::VECCALC> reduceWorkBuffer_;
    TBuf<TPosition::VECCALC> reduceSumBuffer_;
    TBuf<TPosition::VECCALC> inputBuffer_;
    TBuf<TPosition::VECCALC> shiftedBuffer_;
    TBuf<TPosition::VECCALC> accumulatorBuffer_;
    TBuf<TPosition::VECCALC> offsetBuffer_;
    GlobalTensor<float> valuesGm_;
    GlobalTensor<float> correlationGm_;
    GlobalTensor<float> outputGm_;
    uint32_t batch_ = 0U;
    uint32_t heads_ = 0U;
    uint32_t channels_ = 0U;
    uint32_t length_ = 0U;
    uint32_t topK_ = 0U;
    float inverseHeads_ = 1.0F;
    float inverseChannels_ = 1.0F;
};

}  // namespace

extern "C" __global__ __aicore__ void autoformer_inference_aggregate_fused(GM_ADDR values, GM_ADDR correlation,
                                                                           GM_ADDR output, GM_ADDR workspace,
                                                                           GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    const auto* config = reinterpret_cast<const AutocorrTopkTiling*>(&tilingData);
    AutocorrInferenceFusedKernel kernel;
    kernel.Init(values, correlation, output, *config);
    kernel.Process();
}
