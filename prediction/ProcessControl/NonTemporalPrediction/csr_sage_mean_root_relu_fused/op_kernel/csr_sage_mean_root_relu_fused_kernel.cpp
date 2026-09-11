/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct CsrSageMeanRootReluTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t coreNum;
};

class CsrSageMeanRootReluFusedKernel
{
   public:
    __aicore__ inline CsrSageMeanRootReluFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR columnIndex, GM_ADDR neighborFeatures, GM_ADDR rootFeatures,
                                GM_ADDR output, const __gm__ CsrSageMeanRootReluTiling* tiling)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        channels_ = tiling->channels;
        channelStride_ = (channels_ + 7U) / 8U * 8U;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        columnIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(columnIndex), edges_);
        neighborFeaturesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(neighborFeatures),
                                            static_cast<uint64_t>(nodes_) * channels_);
        rootFeaturesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rootFeatures),
                                        static_cast<uint64_t>(nodes_) * channels_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), static_cast<uint64_t>(nodes_) * channels_);
        pipe_.InitBuffer(accumulatorBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(rowBuf_, channelStride_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        LocalTensor<float> row = rowBuf_.Get<float>();
        const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        DataCopyExtParams rowCopy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        for (uint32_t node = GetBlockIdx(); node < nodes_; node += GetBlockNum())
        {
            Duplicate(accumulator, 0.0f, channelStride_);
            pipe_barrier(PIPE_ALL);
            const int32_t begin = rowPtrGm_.GetValue(node);
            const int32_t end = rowPtrGm_.GetValue(node + 1U);
            for (int32_t edge = begin; edge < end; ++edge)
            {
                const uint32_t source = static_cast<uint32_t>(columnIndexGm_.GetValue(edge));
                DataCopyPad(row, neighborFeaturesGm_[static_cast<uint64_t>(source) * channels_], rowCopy, pad);
                pipe_barrier(PIPE_ALL);
                Add(accumulator, accumulator, row, channelStride_);
                pipe_barrier(PIPE_ALL);
            }
            if (end > begin)
            {
                Muls(accumulator, accumulator, 1.0f / static_cast<float>(end - begin), channelStride_);
                pipe_barrier(PIPE_ALL);
            }
            const uint64_t nodeOffset = static_cast<uint64_t>(node) * channels_;
            DataCopyPad(row, rootFeaturesGm_[nodeOffset], rowCopy, pad);
            pipe_barrier(PIPE_ALL);
            Add(accumulator, accumulator, row, channelStride_);
            Maxs(accumulator, accumulator, 0.0f, channelStride_);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(outputGm_[nodeOffset], accumulator, rowCopy);
            pipe_barrier(PIPE_ALL);
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> accumulatorBuf_, rowBuf_;
    GlobalTensor<int32_t> rowPtrGm_, columnIndexGm_;
    GlobalTensor<float> neighborFeaturesGm_, rootFeaturesGm_, outputGm_;
    uint32_t nodes_ = 0U;
    uint32_t edges_ = 0U;
    uint32_t channels_ = 0U;
    uint32_t channelStride_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_sage_mean_root_relu_fused_kernel(GM_ADDR rowPtr, GM_ADDR columnIndex,
                                                                           GM_ADDR neighborFeatures,
                                                                           GM_ADDR rootFeatures, GM_ADDR output,
                                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrSageMeanRootReluFusedKernel op;
    op.Init(rowPtr, columnIndex, neighborFeatures, rootFeatures, output,
            reinterpret_cast<const __gm__ CsrSageMeanRootReluTiling*>(tiling));
    op.Process();
}
