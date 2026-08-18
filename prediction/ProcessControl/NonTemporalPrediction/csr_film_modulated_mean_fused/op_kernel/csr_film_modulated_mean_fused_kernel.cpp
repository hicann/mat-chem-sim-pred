/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct CsrFilmModulatedMeanFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t coreNum;
    uint32_t maxSegmentSize;
    uint32_t applyRelu;
    uint32_t reserved[2];
};

class CsrFilmModulatedMeanFusedKernel
{
   public:
    __aicore__ inline CsrFilmModulatedMeanFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR projected, GM_ADDR beta, GM_ADDR gamma,
                                GM_ADDR output, const __gm__ CsrFilmModulatedMeanFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        channels_ = tiling->channels;
        maxSegmentSize_ = tiling->maxSegmentSize;
        applyRelu_ = tiling->applyRelu;
        channelStride_ = (channels_ + 7U) / 8U * 8U;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        projectedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(projected),
                                     static_cast<uint64_t>(nodes_) * channels_);
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta), static_cast<uint64_t>(nodes_) * channels_);
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gamma), static_cast<uint64_t>(nodes_) * channels_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), static_cast<uint64_t>(nodes_) * channels_);
        pipe_.InitBuffer(outputBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(betaBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(gammaBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(messageBuf_, channelStride_ * sizeof(float));
    }

    __aicore__ inline bool ValidRow(int32_t begin, int32_t end)
    {
        bool valid = begin >= 0 && end >= begin && static_cast<uint32_t>(end) <= edges_ &&
                     static_cast<uint32_t>(end - begin) <= maxSegmentSize_;
        for (int32_t edge = begin; valid && edge < end; ++edge)
        {
            const int32_t source = sourceIndexGm_.GetValue(edge);
            valid = source >= 0 && static_cast<uint32_t>(source) < nodes_;
        }
        return valid;
    }

    __aicore__ inline void ProcessTarget(uint32_t target, LocalTensor<float> aggregate, LocalTensor<float> beta,
                                         LocalTensor<float> gamma, LocalTensor<float> message)
    {
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
        Duplicate(aggregate, 0.0F, channelStride_);
        const int32_t begin = rowPtrGm_.GetValue(target);
        const int32_t end = rowPtrGm_.GetValue(target + 1U);
        const uint32_t count = ValidRow(begin, end) ? static_cast<uint32_t>(end - begin) : 0U;
        const uint64_t targetBase = static_cast<uint64_t>(target) * channels_;
        DataCopyPad(beta, betaGm_[targetBase], copy, pad);
        DataCopyPad(gamma, gammaGm_[targetBase], copy, pad);
        pipe_barrier(PIPE_ALL);
        float countFloat = 0.0F;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(static_cast<uint32_t>(begin) + item));
            DataCopyPad(message, projectedGm_[static_cast<uint64_t>(source) * channels_], copy, pad);
            pipe_barrier(PIPE_ALL);
            Mul(message, message, gamma, channels_);
            Add(message, message, beta, channels_);
            if (applyRelu_ != 0U)
            {
                Relu(message, message, channels_);
            }
            Add(aggregate, aggregate, message, channels_);
            countFloat += 1.0F;
            pipe_barrier(PIPE_ALL);
        }
        if (count > 0U)
        {
            Muls(aggregate, aggregate, 1.0F / countFloat, channels_);
        }
        pipe_barrier(PIPE_ALL);
        DataCopyPad(outputGm_[targetBase], aggregate, copy);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> aggregate = outputBuf_.Get<float>();
        LocalTensor<float> beta = betaBuf_.Get<float>();
        LocalTensor<float> gamma = gammaBuf_.Get<float>();
        LocalTensor<float> message = messageBuf_.Get<float>();
        for (uint32_t target = GetBlockIdx(); target < nodes_; target += GetBlockNum())
        {
            ProcessTarget(target, aggregate, beta, gamma, message);
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> outputBuf_, betaBuf_, gammaBuf_, messageBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<float> projectedGm_, betaGm_, gammaGm_, outputGm_;
    uint32_t nodes_ = 0U, edges_ = 0U, channels_ = 0U;
    uint32_t maxSegmentSize_ = 0U, applyRelu_ = 0U, channelStride_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_film_modulated_mean_fused_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                           GM_ADDR projected, GM_ADDR beta,
                                                                           GM_ADDR gamma, GM_ADDR output,
                                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrFilmModulatedMeanFusedKernel op;
    op.Init(rowPtr, sourceIndex, projected, beta, gamma, output,
            reinterpret_cast<const __gm__ CsrFilmModulatedMeanFusedTiling*>(tiling));
    op.Process();
}
