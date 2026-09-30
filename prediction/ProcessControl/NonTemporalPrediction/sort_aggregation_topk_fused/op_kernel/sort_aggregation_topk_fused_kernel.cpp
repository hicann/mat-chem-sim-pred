/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cfloat>

#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct SortKernelTiling
{
    // Kernel-local top-k layout is intentionally distinct from the host wrapper.
    static constexpr uint32_t kKernelLayoutTag = 0x534F5254U;
    uint32_t node_count;
    uint32_t graph_count;
    uint32_t channel_count;
    uint32_t top_k;
    uint32_t core_count;
    uint32_t reserved[3];
};

class SortAggregationTopkFusedKernel
{
   public:
    __aicore__ inline SortAggregationTopkFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR graphPtr, GM_ADDR features, GM_ADDR output,
                                const __gm__ SortKernelTiling* tiling)
    {
        nodes_ = tiling->node_count;
        graphs_ = tiling->graph_count;
        channels_ = tiling->channel_count;
        topK_ = tiling->top_k;
        channelStride_ = (channels_ + 7U) / 8U * 8U;
        topKStride_ = (topK_ + 7U) / 8U * 8U;
        graphPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(graphPtr), graphs_ + 1U);
        featuresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(features),
                                    static_cast<uint64_t>(nodes_) * channels_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output),
                                  static_cast<uint64_t>(graphs_) * topK_ * channels_);
        pipe_.InitBuffer(scoresBuf_, topKStride_ * sizeof(float));
        pipe_.InitBuffer(indicesBuf_, topKStride_ * sizeof(int32_t));
        pipe_.InitBuffer(rowBuf_, channelStride_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> scores = scoresBuf_.Get<float>();
        LocalTensor<int32_t> indices = indicesBuf_.Get<int32_t>();
        LocalTensor<float> row = rowBuf_.Get<float>();
        const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        for (uint32_t graph = core; graph < graphs_; graph += cores)
        {
            ProcessGraph(graph, scores, indices, row, pad, copy);
        }
    }

   private:
    __aicore__ inline void ProcessGraph(uint32_t graph, LocalTensor<float> scores, LocalTensor<int32_t> indices,
                                        LocalTensor<float> row, const DataCopyPadExtParams<float>& pad,
                                        const DataCopyExtParams& copy)
    {
        for (uint32_t slot = 0U; slot < topK_; ++slot)
        {
            scores.SetValue(slot, -FLT_MAX);
            indices.SetValue(slot, -1);
        }
        const int32_t begin = graphPtrGm_.GetValue(graph);
        const int32_t end = graphPtrGm_.GetValue(graph + 1U);
        for (int32_t node = begin; node < end; ++node)
        {
            const float score = featuresGm_.GetValue(static_cast<uint64_t>(node) * channels_ + channels_ - 1U);
            uint32_t position = topK_;
            for (uint32_t slot = 0U; slot < topK_; ++slot)
            {
                if (score > scores.GetValue(slot))
                {
                    position = slot;
                    break;
                }
            }
            if (position < topK_)
            {
                for (uint32_t slot = topK_ - 1U; slot > position; --slot)
                {
                    scores.SetValue(slot, scores.GetValue(slot - 1U));
                    indices.SetValue(slot, indices.GetValue(slot - 1U));
                }
                scores.SetValue(position, score);
                indices.SetValue(position, node);
            }
        }
        WriteGraph(graph, indices, row, pad, copy);
    }

    __aicore__ inline void WriteGraph(uint32_t graph, LocalTensor<int32_t> indices, LocalTensor<float> row,
                                      const DataCopyPadExtParams<float>& pad, const DataCopyExtParams& copy)
    {
        for (uint32_t slot = 0U; slot < topK_; ++slot)
        {
            const uint64_t outputOffset = (static_cast<uint64_t>(graph) * topK_ + slot) * channels_;
            const int32_t node = indices.GetValue(slot);
            if (node >= 0)
            {
                DataCopyPad(row, featuresGm_[static_cast<uint64_t>(node) * channels_], copy, pad);
            }
            else
            {
                Duplicate(row, 0.0F, channelStride_);
            }
            pipe_barrier(PIPE_ALL);
            DataCopyPad(outputGm_[outputOffset], row, copy);
            pipe_barrier(PIPE_ALL);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> scoresBuf_, indicesBuf_, rowBuf_;
    GlobalTensor<int32_t> graphPtrGm_;
    GlobalTensor<float> featuresGm_, outputGm_;
    uint32_t nodes_ = 0U;
    uint32_t graphs_ = 0U;
    uint32_t channels_ = 0U;
    uint32_t topK_ = 0U;
    uint32_t channelStride_ = 0U;
    uint32_t topKStride_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void sort_aggregation_topk_fused_kernel(GM_ADDR graphPtr, GM_ADDR features,
                                                                         GM_ADDR output, GM_ADDR workspace,
                                                                         GM_ADDR tiling)
{
    (void)workspace;
    SortAggregationTopkFusedKernel op;
    op.Init(graphPtr, features, output, reinterpret_cast<const __gm__ SortKernelTiling*>(tiling));
    op.Process();
}
