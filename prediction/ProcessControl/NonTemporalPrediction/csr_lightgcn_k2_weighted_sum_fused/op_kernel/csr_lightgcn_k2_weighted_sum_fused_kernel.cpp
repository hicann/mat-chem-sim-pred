/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct CsrLightgcnK2WeightedSumFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t coreNum;
    float alpha0;
    float alpha1;
    float alpha2;
    uint32_t reserved;
};

class CsrLightgcnK2WeightedSumFusedKernel
{
   public:
    __aicore__ inline CsrLightgcnK2WeightedSumFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR norm, GM_ADDR features, GM_ADDR output,
                                GM_ADDR temporary, const __gm__ CsrLightgcnK2WeightedSumFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        channels_ = tiling->channels;
        alpha0_ = tiling->alpha0;
        alpha1_ = tiling->alpha1;
        alpha2_ = tiling->alpha2;
        stride_ = (channels_ + 7U) / 8U * 8U;
        plane_ = static_cast<uint64_t>(nodes_) * channels_;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(norm), edges_);
        featuresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(features), plane_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), plane_);
        temporaryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(temporary), plane_);
        pipe_.InitBuffer(accumulatorBuf_, stride_ * sizeof(float));
        pipe_.InitBuffer(messageBuf_, stride_ * sizeof(float));
        pipe_.InitBuffer(featureBuf_, stride_ * sizeof(float));
    }

    __aicore__ inline bool ValidRow(int32_t begin, int32_t end)
    {
        bool valid = begin >= 0 && end >= begin && static_cast<uint32_t>(end) <= edges_;
        for (int32_t edge = begin; valid && edge < end; ++edge)
        {
            const int32_t source = sourceIndexGm_.GetValue(edge);
            valid = source >= 0 && static_cast<uint32_t>(source) < nodes_;
        }
        return valid;
    }

    __aicore__ inline void Aggregate(uint32_t target, bool fromTemporary, LocalTensor<float> accumulator,
                                     LocalTensor<float> message)
    {
        const DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0F};
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        const int32_t begin = rowPtrGm_.GetValue(target);
        const int32_t end = rowPtrGm_.GetValue(target + 1U);
        Duplicate(accumulator, 0.0F, stride_);
        if (!ValidRow(begin, end))
        {
            return;
        }
        for (int32_t edge = begin; edge < end; ++edge)
        {
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(edge));
            Duplicate(message, 0.0F, stride_);
            if (fromTemporary)
            {
                DataCopyPad(message, temporaryGm_[static_cast<uint64_t>(source) * channels_], copy, pad);
            }
            else
            {
                DataCopyPad(message, featuresGm_[static_cast<uint64_t>(source) * channels_], copy, pad);
            }
            pipe_barrier(PIPE_ALL);
            Muls(message, message, normGm_.GetValue(edge), stride_);
            Add(accumulator, accumulator, message, stride_);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void Stage1()
    {
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        LocalTensor<float> message = messageBuf_.Get<float>();
        LocalTensor<float> feature = featureBuf_.Get<float>();
        const DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0F};
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        for (uint32_t target = GetBlockIdx(); target < nodes_; target += GetBlockNum())
        {
            Duplicate(feature, 0.0F, stride_);
            Aggregate(target, false, accumulator, message);
            DataCopyPad(feature, featuresGm_[static_cast<uint64_t>(target) * channels_], copy, pad);
            pipe_barrier(PIPE_ALL);
            Muls(feature, feature, alpha0_, stride_);
            Muls(message, accumulator, alpha1_, stride_);
            Add(feature, feature, message, stride_);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(temporaryGm_[static_cast<uint64_t>(target) * channels_], accumulator, copy);
            DataCopyPad(outputGm_[static_cast<uint64_t>(target) * channels_], feature, copy);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void Stage2()
    {
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        LocalTensor<float> message = messageBuf_.Get<float>();
        LocalTensor<float> feature = featureBuf_.Get<float>();
        const DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0F};
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        for (uint32_t target = GetBlockIdx(); target < nodes_; target += GetBlockNum())
        {
            Aggregate(target, true, accumulator, message);
            Duplicate(feature, 0.0F, stride_);
            DataCopyPad(feature, outputGm_[static_cast<uint64_t>(target) * channels_], copy, pad);
            pipe_barrier(PIPE_ALL);
            Muls(accumulator, accumulator, alpha2_, stride_);
            Add(feature, feature, accumulator, stride_);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(outputGm_[static_cast<uint64_t>(target) * channels_], feature, copy);
            pipe_barrier(PIPE_ALL);
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> accumulatorBuf_, messageBuf_, featureBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<float> normGm_, featuresGm_, outputGm_, temporaryGm_;
    uint32_t nodes_ = 0U, edges_ = 0U, channels_ = 0U, stride_ = 0U;
    uint64_t plane_ = 0U;
    float alpha0_ = 0.0F, alpha1_ = 0.0F, alpha2_ = 0.0F;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_lightgcn_k2_weighted_sum_stage1_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                                 GM_ADDR norm, GM_ADDR features,
                                                                                 GM_ADDR output, GM_ADDR temporary,
                                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrLightgcnK2WeightedSumFusedKernel op;
    op.Init(rowPtr, sourceIndex, norm, features, output, temporary,
            reinterpret_cast<const __gm__ CsrLightgcnK2WeightedSumFusedTiling*>(tiling));
    op.Stage1();
}
extern "C" __global__ __aicore__ void csr_lightgcn_k2_weighted_sum_stage2_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                                 GM_ADDR norm, GM_ADDR features,
                                                                                 GM_ADDR output, GM_ADDR temporary,
                                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrLightgcnK2WeightedSumFusedKernel op;
    op.Init(rowPtr, sourceIndex, norm, features, output, temporary,
            reinterpret_cast<const __gm__ CsrLightgcnK2WeightedSumFusedTiling*>(tiling));
    op.Stage2();
}
