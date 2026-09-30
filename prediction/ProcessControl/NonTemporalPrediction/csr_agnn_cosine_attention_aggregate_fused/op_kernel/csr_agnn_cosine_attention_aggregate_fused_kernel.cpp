/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct CsrAgnnCosineAttentionAggregateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t coreNum;
    uint32_t maxSegmentSize;
    float beta;
    uint32_t reserved0;
    uint32_t reserved1;
};

class CsrAgnnCosineAttentionAggregateFusedKernel
{
   public:
    __aicore__ inline CsrAgnnCosineAttentionAggregateFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR features, GM_ADDR normalized,
                                GM_ADDR output, const __gm__ CsrAgnnCosineAttentionAggregateFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        channels_ = tiling->channels;
        maxSegmentSize_ = tiling->maxSegmentSize;
        beta_ = tiling->beta;
        channelStride_ = (channels_ + 7U) / 8U * 8U;
        segmentStride_ = (maxSegmentSize_ + 7U) / 8U * 8U;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        featuresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(features),
                                    static_cast<uint64_t>(nodes_) * channels_);
        normalizedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normalized),
                                      static_cast<uint64_t>(nodes_) * channels_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), static_cast<uint64_t>(nodes_) * channels_);
        pipe_.InitBuffer(attentionBuf_, segmentStride_ * sizeof(float));
        pipe_.InitBuffer(outputBuf_, channelStride_ * sizeof(float));
        scalarToVectorEvent_ = pipe_.AllocEventID<HardEvent::S_V>();
        vectorToScalarEvent_ = pipe_.AllocEventID<HardEvent::V_S>();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> attention = attentionBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        const DataCopyExtParams outputCopy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        for (uint32_t target = GetBlockIdx(); target < nodes_; target += GetBlockNum())
        {
            Duplicate(output, 0.0F, channelStride_);
            const int32_t begin = rowPtrGm_.GetValue(target);
            const int32_t end = rowPtrGm_.GetValue(target + 1U);
            bool valid = begin >= 0 && end >= begin && static_cast<uint32_t>(end) <= edges_ &&
                         static_cast<uint32_t>(end - begin) <= maxSegmentSize_;
            for (int32_t edge = begin; valid && edge < end; ++edge)
            {
                const int32_t source = sourceIndexGm_.GetValue(edge);
                valid = source >= 0 && static_cast<uint32_t>(source) < nodes_;
            }
            const uint32_t count = valid ? static_cast<uint32_t>(end - begin) : 0U;
            const uint64_t targetBase = static_cast<uint64_t>(target) * channels_;
            const float maximum = ComputeLogits(attention, targetBase, begin, count);
            Aggregate(attention, output, begin, count, maximum);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(outputGm_[targetBase], output, outputCopy);
            pipe_barrier(PIPE_ALL);
        }
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent_);
        pipe_.ReleaseEventID<HardEvent::V_S>(vectorToScalarEvent_);
    }

   private:
    __aicore__ inline float ComputeLogits(LocalTensor<float> attention, uint64_t targetBase, int32_t begin,
                                          uint32_t count)
    {
        float maximum = -3.402823466e38F;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(begin + item));
            const uint64_t sourceBase = static_cast<uint64_t>(source) * channels_;
            float dot = 0.0F;
            for (uint32_t channel = 0U; channel < channels_; ++channel)
            {
                dot += normalizedGm_.GetValue(targetBase + channel) * normalizedGm_.GetValue(sourceBase + channel);
            }
            const float logit = beta_ * dot;
            attention.SetValue(item, logit);
            maximum = logit > maximum ? logit : maximum;
        }
        return maximum;
    }

    __aicore__ inline void Aggregate(LocalTensor<float> attention, LocalTensor<float> output, int32_t begin,
                                     uint32_t count, float maximum)
    {
        if (count > 0U)
        {
            SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
            WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);
            Adds(attention, attention, -maximum, count);
            Exp(attention, attention, count);
            SetFlag<HardEvent::V_S>(vectorToScalarEvent_);
            WaitFlag<HardEvent::V_S>(vectorToScalarEvent_);
            float denominator = 0.0F;
            for (uint32_t item = 0U; item < count; ++item)
            {
                denominator += attention.GetValue(item);
            }
            const float inverse = 1.0F / denominator;
            for (uint32_t item = 0U; item < count; ++item)
            {
                const float weight = attention.GetValue(item) * inverse;
                const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(begin + item));
                const uint64_t sourceBase = static_cast<uint64_t>(source) * channels_;
                for (uint32_t channel = 0U; channel < channels_; ++channel)
                {
                    output.SetValue(channel,
                                    output.GetValue(channel) + weight * featuresGm_.GetValue(sourceBase + channel));
                }
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> attentionBuf_, outputBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<float> featuresGm_, normalizedGm_, outputGm_;
    uint32_t nodes_ = 0U, edges_ = 0U, channels_ = 0U;
    uint32_t maxSegmentSize_ = 0U, channelStride_ = 0U;
    uint32_t segmentStride_ = 0U;
    float beta_ = 1.0F;
    TEventID scalarToVectorEvent_ = 0, vectorToScalarEvent_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_agnn_cosine_attention_aggregate_fused_kernel(
    GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR features, GM_ADDR normalized, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    CsrAgnnCosineAttentionAggregateFusedKernel op;
    op.Init(rowPtr, sourceIndex, features, normalized, output,
            reinterpret_cast<const __gm__ CsrAgnnCosineAttentionAggregateFusedTiling*>(tiling));
    op.Process();
}
