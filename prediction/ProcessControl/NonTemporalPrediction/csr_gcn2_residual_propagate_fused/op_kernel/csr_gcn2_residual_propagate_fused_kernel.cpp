/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "../../spectral_basis_kernel_common.h"

using namespace AscendC;

namespace
{
struct CsrGcn2ResidualPropagateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t coreNum;
    float alpha;
    uint32_t reserved[3];
};

class CsrGcn2ResidualPropagateFusedKernel
{
   public:
    __aicore__ inline CsrGcn2ResidualPropagateFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR edgeWeight, GM_ADDR current,
                                GM_ADDR initial, GM_ADDR output,
                                const __gm__ CsrGcn2ResidualPropagateFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        channels_ = tiling->channels;
        stride_ = (channels_ + 7U) / 8U * 8U;
        alpha_ = tiling->alpha;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        edgeWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(edgeWeight), edges_);
        const uint64_t values = static_cast<uint64_t>(nodes_) * channels_;
        currentGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(current), values);
        initialGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(initial), values);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), values);
        pipe_.InitBuffer(accumulatorBuf_, stride_ * sizeof(float));
        pipe_.InitBuffer(messageBuf_, stride_ * sizeof(float));
        pipe_.InitBuffer(initialBuf_, stride_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        LocalTensor<float> message = messageBuf_.Get<float>();
        LocalTensor<float> initial = initialBuf_.Get<float>();
        const DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0F};
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        for (uint32_t target = GetBlockIdx(); target < nodes_; target += GetBlockNum())
        {
            const int32_t begin = rowPtrGm_.GetValue(target);
            const int32_t end = rowPtrGm_.GetValue(target + 1U);
            const bool valid = SpectralBasis::ValidSources(sourceIndexGm_, begin, end, nodes_, edges_);
            Duplicate(accumulator, 0.0F, stride_);
            if (valid)
            {
                for (int32_t edge = begin; edge < end; ++edge)
                {
                    const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(edge));
                    Duplicate(message, 0.0F, stride_);
                    DataCopyPad(message, currentGm_[static_cast<uint64_t>(source) * channels_], copy, pad);
                    pipe_barrier(PIPE_ALL);
                    Muls(message, message, (1.0F - alpha_) * edgeWeightGm_.GetValue(edge), stride_);
                    Add(accumulator, accumulator, message, stride_);
                    pipe_barrier(PIPE_ALL);
                }
                Duplicate(initial, 0.0F, stride_);
                DataCopyPad(initial, initialGm_[static_cast<uint64_t>(target) * channels_], copy, pad);
                pipe_barrier(PIPE_ALL);
                Muls(initial, initial, alpha_, stride_);
                Add(accumulator, accumulator, initial, stride_);
                pipe_barrier(PIPE_ALL);
            }
            DataCopyPad(outputGm_[static_cast<uint64_t>(target) * channels_], accumulator, copy);
            pipe_barrier(PIPE_ALL);
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> accumulatorBuf_, messageBuf_, initialBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<float> edgeWeightGm_, currentGm_, initialGm_, outputGm_;
    uint32_t nodes_ = 0U, edges_ = 0U, channels_ = 0U, stride_ = 0U;
    float alpha_ = 0.0F;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_gcn2_residual_propagate_fused_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                               GM_ADDR edgeWeight, GM_ADDR current,
                                                                               GM_ADDR initial, GM_ADDR output,
                                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrGcn2ResidualPropagateFusedKernel op;
    op.Init(rowPtr, sourceIndex, edgeWeight, current, initial, output,
            reinterpret_cast<const __gm__ CsrGcn2ResidualPropagateFusedTiling*>(tiling));
    op.Process();
}
