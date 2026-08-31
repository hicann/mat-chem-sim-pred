/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct CsrPointTransformerAttentionAggregateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t maxSegmentSize;
    uint32_t coreNum;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
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
class CsrPointTransformerAttentionAggregateFusedKernel
{
   public:
    __aicore__ inline CsrPointTransformerAttentionAggregateFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR alphaSource, GM_ADDR alphaTarget,
                                GM_ADDR value, GM_ADDR delta, GM_ADDR output,
                                const __gm__ CsrPointTransformerAttentionAggregateFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        channels_ = tiling->channels;
        maxSegmentSize_ = tiling->maxSegmentSize;
        segmentStride_ = (maxSegmentSize_ + 7U) / 8U * 8U;
        channelStride_ = (channels_ + 7U) / 8U * 8U;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        alphaSourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(alphaSource),
                                       static_cast<uint64_t>(nodes_) * channels_);
        alphaTargetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(alphaTarget),
                                       static_cast<uint64_t>(nodes_) * channels_);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), static_cast<uint64_t>(nodes_) * channels_);
        deltaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(delta), static_cast<uint64_t>(edges_) * channels_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), static_cast<uint64_t>(nodes_) * channels_);
        pipe_.InitBuffer(weightBuf_, segmentStride_ * sizeof(float));
        pipe_.InitBuffer(outputBuf_, channelStride_ * sizeof(float));
        if constexpr (!std::is_same_v<T, float>)
        {
            pipe_.InitBuffer(outputTypeBuf_, channelStride_ * sizeof(T));
        }
        scalarToVectorEvent_ = pipe_.AllocEventID<HardEvent::S_V>();
        vectorToScalarEvent_ = pipe_.AllocEventID<HardEvent::V_S>();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> weights = weightBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        for (uint32_t target = GetBlockIdx(); target < nodes_; target += GetBlockNum())
        {
            ProcessTarget(target, weights, output);
        }
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent_);
        pipe_.ReleaseEventID<HardEvent::V_S>(vectorToScalarEvent_);
    }

   private:
    __aicore__ inline uint32_t SegmentSize(uint32_t target, int32_t& begin)
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

    __aicore__ inline float FillWeights(int32_t begin, uint32_t count, uint64_t targetBase, uint32_t channel,
                                        LocalTensor<float> weights)
    {
        float maximum = -3.402823466e38F;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t edge = static_cast<uint32_t>(begin) + item;
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(edge));
            const float score =
                ToCompute(alphaTargetGm_.GetValue(targetBase + channel)) -
                ToCompute(alphaSourceGm_.GetValue(static_cast<uint64_t>(source) * channels_ + channel)) +
                ToCompute(deltaGm_.GetValue(static_cast<uint64_t>(edge) * channels_ + channel));
            weights.SetValue(item, score);
            maximum = score > maximum ? score : maximum;
        }
        return maximum;
    }

    __aicore__ inline float NormalizeWeights(uint32_t count, float maximum, LocalTensor<float> weights)
    {
        SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
        WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);
        Adds(weights, weights, -maximum, count);
        Exp(weights, weights, count);
        SetFlag<HardEvent::V_S>(vectorToScalarEvent_);
        WaitFlag<HardEvent::V_S>(vectorToScalarEvent_);
        float denominator = 0.0F;
        for (uint32_t item = 0U; item < count; ++item)
        {
            denominator += weights.GetValue(item);
        }
        return denominator > 0.0F ? 1.0F / denominator : 0.0F;
    }

    __aicore__ inline float Aggregate(int32_t begin, uint32_t count, uint32_t channel, float inverse,
                                      LocalTensor<float> weights)
    {
        float aggregate = 0.0F;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t edge = static_cast<uint32_t>(begin) + item;
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(edge));
            const float message = ToCompute(valueGm_.GetValue(static_cast<uint64_t>(source) * channels_ + channel)) +
                                  ToCompute(deltaGm_.GetValue(static_cast<uint64_t>(edge) * channels_ + channel));
            aggregate += weights.GetValue(item) * inverse * message;
        }
        return aggregate;
    }

    __aicore__ inline void WriteOutput(uint64_t targetBase, LocalTensor<float> output)
    {
        pipe_barrier(PIPE_ALL);
        if constexpr (std::is_same_v<T, float>)
        {
            const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(outputGm_[targetBase], output, copy);
        }
        else
        {
            LocalTensor<T> outputTyped = outputTypeBuf_.Get<T>();
            Cast(outputTyped, output, RoundMode::CAST_RINT, channels_);
            pipe_barrier(PIPE_ALL);
            const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm_[targetBase], outputTyped, copy);
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ProcessTarget(uint32_t target, LocalTensor<float> weights, LocalTensor<float> output)
    {
        Duplicate(output, 0.0F, channelStride_);
        int32_t begin = 0;
        const uint32_t count = SegmentSize(target, begin);
        const uint64_t targetBase = static_cast<uint64_t>(target) * channels_;
        for (uint32_t channel = 0U; channel < channels_ && count > 0U; ++channel)
        {
            const float maximum = FillWeights(begin, count, targetBase, channel, weights);
            const float inverse = NormalizeWeights(count, maximum, weights);
            output.SetValue(channel, Aggregate(begin, count, channel, inverse, weights));
        }
        WriteOutput(targetBase, output);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> weightBuf_, outputBuf_, outputTypeBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<T> alphaSourceGm_, alphaTargetGm_, valueGm_, deltaGm_, outputGm_;
    uint32_t nodes_ = 0U, edges_ = 0U, channels_ = 0U;
    uint32_t maxSegmentSize_ = 0U, segmentStride_ = 0U, channelStride_ = 0U;
    TEventID scalarToVectorEvent_ = 0U, vectorToScalarEvent_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_point_transformer_attention_aggregate_fused_kernel_fp32(
    GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR alphaSource, GM_ADDR alphaTarget, GM_ADDR value, GM_ADDR delta,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrPointTransformerAttentionAggregateFusedKernel<float> op;
    op.Init(rowPtr, sourceIndex, alphaSource, alphaTarget, value, delta, output,
            reinterpret_cast<const __gm__ CsrPointTransformerAttentionAggregateFusedTiling*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void csr_point_transformer_attention_aggregate_fused_kernel_fp16(
    GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR alphaSource, GM_ADDR alphaTarget, GM_ADDR value, GM_ADDR delta,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrPointTransformerAttentionAggregateFusedKernel<half> op;
    op.Init(rowPtr, sourceIndex, alphaSource, alphaTarget, value, delta, output,
            reinterpret_cast<const __gm__ CsrPointTransformerAttentionAggregateFusedTiling*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void csr_point_transformer_attention_aggregate_fused_kernel_bf16(
    GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR alphaSource, GM_ADDR alphaTarget, GM_ADDR value, GM_ADDR delta,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrPointTransformerAttentionAggregateFusedKernel<bfloat16_t> op;
    op.Init(rowPtr, sourceIndex, alphaSource, alphaTarget, value, delta, output,
            reinterpret_cast<const __gm__ CsrPointTransformerAttentionAggregateFusedTiling*>(tiling));
    op.Process();
}
