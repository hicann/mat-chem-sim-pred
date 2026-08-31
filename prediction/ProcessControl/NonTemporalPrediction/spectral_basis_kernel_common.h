/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#pragma once

#include "kernel_operator.h"

namespace SpectralBasis
{
struct Tiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t coreNum;
};

__aicore__ inline bool ValidSources(AscendC::GlobalTensor<int32_t>& sourceIndex, int32_t begin, int32_t end,
                                    uint32_t nodes, uint32_t edges)
{
    bool valid = begin >= 0 && end >= begin && static_cast<uint32_t>(end) <= edges;
    for (int32_t edge = begin; valid && edge < end; ++edge)
    {
        const int32_t source = sourceIndex.GetValue(edge);
        valid = source >= 0 && static_cast<uint32_t>(source) < nodes;
    }
    return valid;
}

class Kernel
{
   public:
    __aicore__ inline Kernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR norm, GM_ADDR features, GM_ADDR basis,
                                const __gm__ Tiling* tiling, uint32_t planes)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        channels_ = tiling->channels;
        stride_ = (channels_ + 7U) / 8U * 8U;
        plane_ = static_cast<uint64_t>(nodes_) * channels_;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        normGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(norm), edges_);
        featuresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(features), plane_);
        basisGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(basis), static_cast<uint64_t>(planes) * plane_);
        pipe_.InitBuffer(accumulatorBuf_, stride_ * sizeof(float));
        pipe_.InitBuffer(messageBuf_, stride_ * sizeof(float));
        pipe_.InitBuffer(featureBuf_, stride_ * sizeof(float));
    }

    __aicore__ inline void Stage1()
    {
        AscendC::LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        AscendC::LocalTensor<float> message = messageBuf_.Get<float>();
        AscendC::LocalTensor<float> feature = featureBuf_.Get<float>();
        const AscendC::DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0F};
        const AscendC::DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        for (uint32_t target = AscendC::GetBlockIdx(); target < nodes_; target += AscendC::GetBlockNum())
        {
            int32_t begin = 0;
            int32_t end = 0;
            AscendC::Duplicate(feature, 0.0F, stride_);
            AscendC::Duplicate(accumulator, 0.0F, stride_);
            if (!RowBounds(target, begin, end))
            {
                pipe_barrier(PIPE_ALL);
                StorePlane(0U, target, feature, copy);
                StorePlane(1U, target, accumulator, copy);
                continue;
            }
            AscendC::DataCopyPad(feature, featuresGm_[static_cast<uint64_t>(target) * channels_], copy, pad);
            pipe_barrier(PIPE_ALL);
            StorePlane(0U, target, feature, copy);
            Accumulate(begin, end, 0U, false, accumulator, message, copy, pad);
            StorePlane(1U, target, accumulator, copy);
        }
    }

    __aicore__ inline void Propagate(uint32_t inputPlane, uint32_t outputPlane)
    {
        AscendC::LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        AscendC::LocalTensor<float> message = messageBuf_.Get<float>();
        const AscendC::DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0F};
        const AscendC::DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        for (uint32_t target = AscendC::GetBlockIdx(); target < nodes_; target += AscendC::GetBlockNum())
        {
            int32_t begin = 0;
            int32_t end = 0;
            AscendC::Duplicate(accumulator, 0.0F, stride_);
            if (RowBounds(target, begin, end))
            {
                Accumulate(begin, end, inputPlane, true, accumulator, message, copy, pad);
            }
            StorePlane(outputPlane, target, accumulator, copy);
        }
    }

    __aicore__ inline void ChebyshevStage2()
    {
        AscendC::LocalTensor<float> feature = featureBuf_.Get<float>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        AscendC::LocalTensor<float> message = messageBuf_.Get<float>();
        const AscendC::DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0F};
        for (uint32_t node = AscendC::GetBlockIdx(); node < nodes_; node += AscendC::GetBlockNum())
        {
            int32_t begin = 0, end = 0;
            AscendC::Duplicate(accumulator, 0.0F, stride_);
            if (RowBounds(node, begin, end))
            {
                Accumulate(begin, end, 1U, true, accumulator, message, copy, pad);
                AscendC::Duplicate(feature, 0.0F, stride_);
                AscendC::DataCopyPad(feature, featuresGm_[static_cast<uint64_t>(node) * channels_], copy, pad);
                pipe_barrier(PIPE_ALL);
                AscendC::Muls(accumulator, accumulator, 2.0F, stride_);
                AscendC::Sub(accumulator, accumulator, feature, stride_);
                pipe_barrier(PIPE_ALL);
            }
            StorePlane(2U, node, accumulator, copy);
        }
    }

   private:
    __aicore__ inline bool RowBounds(uint32_t target, int32_t& begin, int32_t& end)
    {
        begin = rowPtrGm_.GetValue(target);
        end = rowPtrGm_.GetValue(target + 1U);
        return ValidSources(sourceIndexGm_, begin, end, nodes_, edges_);
    }

    __aicore__ inline void Accumulate(int32_t begin, int32_t end, uint32_t inputPlane, bool useBasis,
                                      AscendC::LocalTensor<float>& accumulator, AscendC::LocalTensor<float>& message,
                                      const AscendC::DataCopyExtParams& copy,
                                      const AscendC::DataCopyPadExtParams<float>& pad)
    {
        for (int32_t edge = begin; edge < end; ++edge)
        {
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(edge));
            const uint64_t offset = static_cast<uint64_t>(source) * channels_;
            AscendC::Duplicate(message, 0.0F, stride_);
            if (useBasis)
            {
                AscendC::DataCopyPad(message, basisGm_[static_cast<uint64_t>(inputPlane) * plane_ + offset], copy, pad);
            }
            else
            {
                AscendC::DataCopyPad(message, featuresGm_[offset], copy, pad);
            }
            pipe_barrier(PIPE_ALL);
            AscendC::Muls(message, message, normGm_.GetValue(edge), stride_);
            AscendC::Add(accumulator, accumulator, message, stride_);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void StorePlane(uint32_t plane, uint32_t target, AscendC::LocalTensor<float>& value,
                                      const AscendC::DataCopyExtParams& copy)
    {
        const uint64_t offset = static_cast<uint64_t>(plane) * plane_ + static_cast<uint64_t>(target) * channels_;
        AscendC::DataCopyPad(basisGm_[offset], value, copy);
        pipe_barrier(PIPE_ALL);
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulatorBuf_, messageBuf_, featureBuf_;
    AscendC::GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    AscendC::GlobalTensor<float> normGm_, featuresGm_, basisGm_;
    uint32_t nodes_ = 0U, edges_ = 0U, channels_ = 0U, stride_ = 0U;
    uint64_t plane_ = 0U;
};

__aicore__ inline void Run(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR norm, GM_ADDR features, GM_ADDR basis,
                           GM_ADDR tiling, uint32_t planes, uint32_t stage)
{
    Kernel kernel;
    kernel.Init(rowPtr, sourceIndex, norm, features, basis, reinterpret_cast<const __gm__ Tiling*>(tiling), planes);
    if (stage == 0U)
    {
        kernel.Stage1();
    }
    else if (stage == 1U)
    {
        kernel.ChebyshevStage2();
    }
    else
    {
        kernel.Propagate(stage - 1U, stage);
    }
}
}  // namespace SpectralBasis
