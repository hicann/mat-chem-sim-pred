/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct CsrSuperGatKernelTiling
{
    static constexpr uint32_t kKernelLayoutTag = 0x53555045U;
    uint32_t node_count;
    uint32_t edge_count;
    uint32_t head_count;
    uint32_t channel_count;
    uint32_t max_segment_size;
    uint32_t core_count;
    float inverseScale;
    float negativeSlope;
};

template <typename T>
__aicore__ inline float ToCompute(T value)
{
    return static_cast<float>(value);
}

template <>
__aicore__ inline float ToCompute<bfloat16_t>(bfloat16_t value)
{
    return ToFloat(value);
}

template <typename T>
__aicore__ inline T FromCompute(float value)
{
    return static_cast<T>(value);
}

template <>
__aicore__ inline bfloat16_t FromCompute<bfloat16_t>(float value)
{
    return ToBfloat16(value);
}

template <typename T>
class CsrSuperGatDotAttentionAggregateFusedKernel
{
   public:
    __aicore__ inline CsrSuperGatDotAttentionAggregateFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR projected, GM_ADDR output,
                                const __gm__ CsrSuperGatKernelTiling* tiling)
    {
        nodes_ = tiling->node_count;
        edges_ = tiling->edge_count;
        heads_ = tiling->head_count;
        channels_ = tiling->channel_count;
        maxSegmentSize_ = tiling->max_segment_size;
        inverseScale_ = tiling->inverseScale;
        negativeSlope_ = tiling->negativeSlope;
        segmentStride_ = (maxSegmentSize_ + 7U) / 8U * 8U;
        outputValues_ = heads_ * channels_;
        outputStride_ = (outputValues_ + 7U) / 8U * 8U;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        projectedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(projected),
                                     static_cast<uint64_t>(nodes_) * outputValues_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), static_cast<uint64_t>(nodes_) * outputValues_);
        pipe_.InitBuffer(weightBuf_, segmentStride_ * sizeof(float));
        pipe_.InitBuffer(outputBuf_, outputStride_ * sizeof(float));
        if constexpr (!std::is_same_v<T, float>)
        {
            pipe_.InitBuffer(outputTypeBuf_, outputStride_ * sizeof(T));
        }
        scalarToVectorEvent_ = pipe_.AllocEventID<HardEvent::S_V>();
        vectorToScalarEvent_ = pipe_.AllocEventID<HardEvent::V_S>();
    }

    __aicore__ inline void Process() { ProcessCore(); }

    __aicore__ inline uint32_t ValidateSegment(uint32_t target, int32_t& begin)
    {
        begin = rowPtrGm_.GetValue(target);
        const int32_t end = rowPtrGm_.GetValue(target + 1U);
        bool valid = begin >= 0 && end >= begin && static_cast<uint32_t>(end) <= edges_ &&
                     static_cast<uint32_t>(end - begin) <= maxSegmentSize_;
        for (int32_t edge = begin; valid && edge < end; ++edge)
        {
            const int32_t source = sourceIndexGm_.GetValue(edge);
            valid = source >= 0 && static_cast<uint32_t>(source) < nodes_;
        }
        return valid ? static_cast<uint32_t>(end - begin) : 0U;
    }

    __aicore__ inline float ComputeHeadScores(uint32_t target, uint32_t head, int32_t begin, uint32_t count,
                                              LocalTensor<float>& weights)
    {
        float maximum = -3.402823466e38F;
        const uint64_t targetBase = (static_cast<uint64_t>(target) * heads_ + head) * channels_;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(begin + item));
            const uint64_t sourceBase = (static_cast<uint64_t>(source) * heads_ + head) * channels_;
            float score = 0.0F;
            for (uint32_t channel = 0U; channel < channels_; ++channel)
            {
                score += ToCompute(projectedGm_.GetValue(targetBase + channel)) *
                         ToCompute(projectedGm_.GetValue(sourceBase + channel));
            }
            score *= inverseScale_;
            score = score >= 0.0F ? score : score * negativeSlope_;
            weights.SetValue(item, score);
            maximum = score > maximum ? score : maximum;
        }
        return maximum;
    }

    __aicore__ inline void AccumulateHead(uint32_t head, int32_t begin, uint32_t count,
                                          LocalTensor<float>& weights, LocalTensor<float>& output)
    {
        float denominator = 0.0F;
        for (uint32_t item = 0U; item < count; ++item)
        {
            denominator += weights.GetValue(item);
        }
        const float inverse = denominator > 0.0F ? 1.0F / denominator : 0.0F;
        const uint32_t destination = head * channels_;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const float weight = weights.GetValue(item) * inverse;
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(begin + item));
            const uint64_t sourceBase = (static_cast<uint64_t>(source) * heads_ + head) * channels_;
            for (uint32_t channel = 0U; channel < channels_; ++channel)
            {
                const uint32_t offset = destination + channel;
                output.SetValue(offset, output.GetValue(offset) +
                                            weight * ToCompute(projectedGm_.GetValue(sourceBase + channel)));
            }
        }
    }

    __aicore__ inline void WriteOutput(uint32_t target, LocalTensor<float>& output)
    {
        const uint64_t outputBase = static_cast<uint64_t>(target) * outputValues_;
        if constexpr (std::is_same_v<T, float>)
        {
            const DataCopyExtParams outputCopy{1U, static_cast<uint32_t>(outputValues_ * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(outputGm_[outputBase], output, outputCopy);
        }
        else
        {
            LocalTensor<T> outputTyped = outputTypeBuf_.Get<T>();
            Cast(outputTyped, output, RoundMode::CAST_RINT, outputValues_);
            pipe_barrier(PIPE_ALL);
            const DataCopyExtParams outputCopy{1U, static_cast<uint32_t>(outputValues_ * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm_[outputBase], outputTyped, outputCopy);
        }
    }

    __aicore__ inline void ProcessCore()
    {
        LocalTensor<float> weights = weightBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        for (uint32_t target = GetBlockIdx(); target < nodes_; target += GetBlockNum())
        {
            Duplicate(output, 0.0F, outputStride_);
            int32_t begin = 0;
            const uint32_t count = ValidateSegment(target, begin);
            for (uint32_t head = 0U; head < heads_ && count > 0U; ++head)
            {
                const float maximum = ComputeHeadScores(target, head, begin, count, weights);
                SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
                WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);
                Adds(weights, weights, -maximum, count);
                Exp(weights, weights, count);
                SetFlag<HardEvent::V_S>(vectorToScalarEvent_);
                WaitFlag<HardEvent::V_S>(vectorToScalarEvent_);
                AccumulateHead(head, begin, count, weights, output);
            }
            pipe_barrier(PIPE_ALL);
            WriteOutput(target, output);
            pipe_barrier(PIPE_ALL);
        }
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent_);
        pipe_.ReleaseEventID<HardEvent::V_S>(vectorToScalarEvent_);
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> weightBuf_, outputBuf_, outputTypeBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<T> projectedGm_, outputGm_;
    uint32_t nodes_ = 0U, edges_ = 0U, heads_ = 0U, channels_ = 0U;
    uint32_t maxSegmentSize_ = 0U, segmentStride_ = 0U;
    uint32_t outputValues_ = 0U, outputStride_ = 0U;
    float inverseScale_ = 1.0F, negativeSlope_ = 0.2F;
    TEventID scalarToVectorEvent_ = 0U, vectorToScalarEvent_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_supergat_dot_attention_aggregate_fused_kernel_fp32(
    GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR projected, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrSuperGatDotAttentionAggregateFusedKernel<float> op;
    op.Init(rowPtr, sourceIndex, projected, output, reinterpret_cast<const __gm__ CsrSuperGatKernelTiling*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void csr_supergat_dot_attention_aggregate_fused_kernel_fp16(
    GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR projected, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrSuperGatDotAttentionAggregateFusedKernel<half> op;
    op.Init(rowPtr, sourceIndex, projected, output, reinterpret_cast<const __gm__ CsrSuperGatKernelTiling*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void csr_supergat_dot_attention_aggregate_fused_kernel_bf16(
    GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR projected, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrSuperGatDotAttentionAggregateFusedKernel<bfloat16_t> op;
    op.Init(rowPtr, sourceIndex, projected, output, reinterpret_cast<const __gm__ CsrSuperGatKernelTiling*>(tiling));
    op.Process();
}
