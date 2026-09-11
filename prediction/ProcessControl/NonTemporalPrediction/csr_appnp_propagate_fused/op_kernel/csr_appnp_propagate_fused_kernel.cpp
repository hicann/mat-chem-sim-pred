/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../op_host/csr_appnp_propagate_fused_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
class CsrAppnpPropagationStepKernel
{
   public:
    __aicore__ inline CsrAppnpPropagationStepKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR columnIndex, GM_ADDR edgeWeight, GM_ADDR current,
                                GM_ADDR initial, GM_ADDR output, const __gm__ CsrAppnpPropagateFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        channels_ = tiling->channels;
        channelStride_ = (channels_ + 7U) / 8U * 8U;
        alpha_ = tiling->alpha;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        columnIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(columnIndex), edges_);
        edgeWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(edgeWeight), edges_);
        currentGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(current), static_cast<uint64_t>(nodes_) * channels_);
        initialGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(initial), static_cast<uint64_t>(nodes_) * channels_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), static_cast<uint64_t>(nodes_) * channels_);
        pipe_.InitBuffer(accumulatorBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(rowBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(initialBuf_, channelStride_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        LocalTensor<float> row = rowBuf_.Get<float>();
        LocalTensor<float> initial = initialBuf_.Get<float>();
        const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        for (uint32_t node = core; node < nodes_; node += cores)
        {
            Duplicate(accumulator, 0.0f, channelStride_);
            pipe_barrier(PIPE_ALL);
            const int32_t begin = rowPtrGm_.GetValue(node);
            const int32_t end = rowPtrGm_.GetValue(node + 1U);
            for (int32_t edge = begin; edge < end; ++edge)
            {
                const uint32_t source = static_cast<uint32_t>(columnIndexGm_.GetValue(edge));
                DataCopyPad(row, currentGm_[static_cast<uint64_t>(source) * channels_], copy, pad);
                pipe_barrier(PIPE_ALL);
                Muls(row, row, (1.0f - alpha_) * edgeWeightGm_.GetValue(edge), channelStride_);
                pipe_barrier(PIPE_ALL);
                Add(accumulator, accumulator, row, channelStride_);
                pipe_barrier(PIPE_ALL);
            }
            const uint64_t offset = static_cast<uint64_t>(node) * channels_;
            DataCopyPad(initial, initialGm_[offset], copy, pad);
            pipe_barrier(PIPE_ALL);
            Muls(initial, initial, alpha_, channelStride_);
            Add(accumulator, accumulator, initial, channelStride_);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(outputGm_[offset], accumulator, copy);
            pipe_barrier(PIPE_ALL);
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> accumulatorBuf_, rowBuf_, initialBuf_;
    GlobalTensor<int32_t> rowPtrGm_, columnIndexGm_;
    GlobalTensor<float> edgeWeightGm_, currentGm_, initialGm_, outputGm_;
    uint32_t nodes_ = 0U;
    uint32_t edges_ = 0U;
    uint32_t channels_ = 0U;
    uint32_t channelStride_ = 0U;
    float alpha_ = 0.0f;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_appnp_propagate_fused_kernel(GM_ADDR rowPtr, GM_ADDR columnIndex,
                                                                       GM_ADDR edgeWeight, GM_ADDR current,
                                                                       GM_ADDR initial, GM_ADDR output,
                                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrAppnpPropagationStepKernel op;
    op.Init(rowPtr, columnIndex, edgeWeight, current, initial, output,
            reinterpret_cast<const __gm__ CsrAppnpPropagateFusedTiling*>(tiling));
    op.Process();
}
