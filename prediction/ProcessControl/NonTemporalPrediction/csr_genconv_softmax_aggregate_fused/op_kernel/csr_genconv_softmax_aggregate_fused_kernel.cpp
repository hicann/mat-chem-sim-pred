/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
constexpr float kMessageEpsilon = 1.0e-7F;
constexpr float kLowestFloat = -3.402823466e38F;

struct CsrGenConvSoftmaxAggregateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t coreNum;
    float temperature;
};

class CsrGenConvSoftmaxAggregateFusedKernel
{
   public:
    __aicore__ inline CsrGenConvSoftmaxAggregateFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR features, GM_ADDR output,
                                const __gm__ CsrGenConvSoftmaxAggregateFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        channels_ = tiling->channels;
        temperature_ = tiling->temperature;
        channelStride_ = (channels_ + 7U) / 8U * 8U;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        featuresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(features),
                                    static_cast<uint64_t>(nodes_) * channels_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), static_cast<uint64_t>(nodes_) * channels_);
        pipe_.InitBuffer(messageBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(logitBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(maximumBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(denominatorBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(numeratorBuf_, channelStride_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        Buffers buffers{messageBuf_.Get<float>(), logitBuf_.Get<float>(), maximumBuf_.Get<float>(),
                        denominatorBuf_.Get<float>(), numeratorBuf_.Get<float>()};
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        const uint32_t cores = GetBlockNum();
        for (uint32_t target = GetBlockIdx(); target < nodes_; target += cores)
        {
            Duplicate(buffers.numerator, 0.0F, channelStride_);
            const int32_t begin = rowPtrGm_.GetValue(target);
            const int32_t end = rowPtrGm_.GetValue(target + 1U);
            if (ValidRow(begin, end) && begin != end)
            {
                FindMaximum(begin, end, buffers);
                Accumulate(begin, end, buffers);
                Div(buffers.numerator, buffers.numerator, buffers.denominator, channelStride_);
            }
            pipe_barrier(PIPE_ALL);
            DataCopyPad(outputGm_[static_cast<uint64_t>(target) * channels_], buffers.numerator, copy);
            pipe_barrier(PIPE_ALL);
        }
    }

   private:
    struct Buffers
    {
        LocalTensor<float> message;
        LocalTensor<float> logit;
        LocalTensor<float> maximum;
        LocalTensor<float> denominator;
        LocalTensor<float> numerator;
    };

    __aicore__ inline bool ValidRow(int32_t begin, int32_t end)
    {
        if (begin < 0 || end < begin || static_cast<uint32_t>(end) > edges_)
        {
            return false;
        }
        for (int32_t edge = begin; edge < end; ++edge)
        {
            const int32_t source = sourceIndexGm_.GetValue(edge);
            if (source < 0 || static_cast<uint32_t>(source) >= nodes_)
            {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void LoadMessage(int32_t edge, Buffers& buffers)
    {
        const DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0F};
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(edge));
        Duplicate(buffers.message, 0.0F, channelStride_);
        DataCopyPad(buffers.message, featuresGm_[static_cast<uint64_t>(source) * channels_], copy, pad);
        pipe_barrier(PIPE_ALL);
        Relu(buffers.message, buffers.message, channelStride_);
        Adds(buffers.message, buffers.message, kMessageEpsilon, channelStride_);
        Muls(buffers.logit, buffers.message, temperature_, channelStride_);
    }

    __aicore__ inline void FindMaximum(int32_t begin, int32_t end, Buffers& buffers)
    {
        Duplicate(buffers.maximum, kLowestFloat, channelStride_);
        for (int32_t edge = begin; edge < end; ++edge)
        {
            LoadMessage(edge, buffers);
            Max(buffers.maximum, buffers.maximum, buffers.logit, channelStride_);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void Accumulate(int32_t begin, int32_t end, Buffers& buffers)
    {
        Duplicate(buffers.denominator, 0.0F, channelStride_);
        Duplicate(buffers.numerator, 0.0F, channelStride_);
        for (int32_t edge = begin; edge < end; ++edge)
        {
            LoadMessage(edge, buffers);
            Sub(buffers.logit, buffers.logit, buffers.maximum, channelStride_);
            Exp(buffers.logit, buffers.logit, channelStride_);
            Add(buffers.denominator, buffers.denominator, buffers.logit, channelStride_);
            Mul(buffers.logit, buffers.logit, buffers.message, channelStride_);
            Add(buffers.numerator, buffers.numerator, buffers.logit, channelStride_);
            pipe_barrier(PIPE_ALL);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> messageBuf_, logitBuf_, maximumBuf_;
    TBuf<TPosition::VECCALC> denominatorBuf_, numeratorBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<float> featuresGm_, outputGm_;
    uint32_t nodes_ = 0U;
    uint32_t edges_ = 0U;
    uint32_t channels_ = 0U;
    uint32_t channelStride_ = 0U;
    float temperature_ = 1.0F;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_genconv_softmax_aggregate_fused_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                                 GM_ADDR features, GM_ADDR output,
                                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrGenConvSoftmaxAggregateFusedKernel op;
    op.Init(rowPtr, sourceIndex, features, output,
            reinterpret_cast<const __gm__ CsrGenConvSoftmaxAggregateFusedTiling*>(tiling));
    op.Process();
}
