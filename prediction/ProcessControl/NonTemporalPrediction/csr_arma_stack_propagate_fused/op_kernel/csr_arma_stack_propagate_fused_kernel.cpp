/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct CsrArmaStackPropagateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t stacks;
    uint32_t channels;
    uint32_t coreNum;
    uint32_t relu;
    uint32_t reserved[2];
};

class CsrArmaStackPropagateFusedKernel
{
   public:
    __aicore__ inline CsrArmaStackPropagateFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR edgeWeight, GM_ADDR projected,
                                GM_ADDR root, GM_ADDR bias, GM_ADDR output,
                                const __gm__ CsrArmaStackPropagateFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        stacks_ = tiling->stacks;
        channels_ = tiling->channels;
        relu_ = tiling->relu;
        stride_ = (channels_ + 7U) / 8U * 8U;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        edgeWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(edgeWeight), edges_);
        const uint64_t values = static_cast<uint64_t>(stacks_) * nodes_ * channels_;
        projectedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(projected), values);
        rootGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(root), values);
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), static_cast<uint64_t>(stacks_) * channels_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), values);
        pipe_.InitBuffer(accumulatorBuf_, stride_ * sizeof(float));
        pipe_.InitBuffer(valueBuf_, stride_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        LocalTensor<float> value = valueBuf_.Get<float>();
        const DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0F};
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        const uint32_t workItems = stacks_ * nodes_;
        for (uint32_t item = GetBlockIdx(); item < workItems; item += GetBlockNum())
        {
            const uint32_t stack = item / nodes_;
            const uint32_t target = item - stack * nodes_;
            const int32_t begin = rowPtrGm_.GetValue(target);
            const int32_t end = rowPtrGm_.GetValue(target + 1U);
            Duplicate(accumulator, 0.0F, stride_);
            if (ValidRow(begin, end))
            {
                Aggregate(stack, begin, end, accumulator, value, copy, pad);
                AddRootBias(stack, target, accumulator, value, copy, pad);
            }
            Store(stack, target, accumulator, copy);
        }
    }

   private:
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

    __aicore__ inline void Aggregate(uint32_t stack, int32_t begin, int32_t end, LocalTensor<float>& accumulator,
                                     LocalTensor<float>& value, const DataCopyExtParams& copy,
                                     const DataCopyPadExtParams<float>& pad)
    {
        for (int32_t edge = begin; edge < end; ++edge)
        {
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(edge));
            const uint64_t offset = (static_cast<uint64_t>(stack) * nodes_ + source) * channels_;
            Duplicate(value, 0.0F, stride_);
            DataCopyPad(value, projectedGm_[offset], copy, pad);
            pipe_barrier(PIPE_ALL);
            Muls(value, value, edgeWeightGm_.GetValue(edge), stride_);
            Add(accumulator, accumulator, value, stride_);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void AddRootBias(uint32_t stack, uint32_t target, LocalTensor<float>& accumulator,
                                       LocalTensor<float>& value, const DataCopyExtParams& copy,
                                       const DataCopyPadExtParams<float>& pad)
    {
        const uint64_t targetOffset = (static_cast<uint64_t>(stack) * nodes_ + target) * channels_;
        Duplicate(value, 0.0F, stride_);
        DataCopyPad(value, rootGm_[targetOffset], copy, pad);
        pipe_barrier(PIPE_ALL);
        Add(accumulator, accumulator, value, stride_);
        Duplicate(value, 0.0F, stride_);
        DataCopyPad(value, biasGm_[static_cast<uint64_t>(stack) * channels_], copy, pad);
        pipe_barrier(PIPE_ALL);
        Add(accumulator, accumulator, value, stride_);
        if (relu_ != 0U)
        {
            Maxs(accumulator, accumulator, 0.0F, stride_);
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void Store(uint32_t stack, uint32_t target, LocalTensor<float>& accumulator,
                                 const DataCopyExtParams& copy)
    {
        const uint64_t offset = (static_cast<uint64_t>(stack) * nodes_ + target) * channels_;
        DataCopyPad(outputGm_[offset], accumulator, copy);
        pipe_barrier(PIPE_ALL);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> accumulatorBuf_, valueBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<float> edgeWeightGm_, projectedGm_, rootGm_, biasGm_;
    GlobalTensor<float> outputGm_;
    uint32_t nodes_ = 0U, edges_ = 0U, stacks_ = 0U, channels_ = 0U;
    uint32_t stride_ = 0U, relu_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_arma_stack_propagate_fused_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                            GM_ADDR edgeWeight, GM_ADDR projected,
                                                                            GM_ADDR root, GM_ADDR bias, GM_ADDR output,
                                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrArmaStackPropagateFusedKernel op;
    op.Init(rowPtr, sourceIndex, edgeWeight, projected, root, bias, output,
            reinterpret_cast<const __gm__ CsrArmaStackPropagateFusedTiling*>(tiling));
    op.Process();
}
