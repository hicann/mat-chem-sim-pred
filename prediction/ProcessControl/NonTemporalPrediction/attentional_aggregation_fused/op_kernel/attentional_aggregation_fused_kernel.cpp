/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct AttentionKernelTiling
{
    // Kernel-local attention layout is intentionally distinct from the host wrapper.
    static constexpr uint32_t kKernelLayoutTag = 0x4154544EU;
    uint32_t node_count;
    uint32_t graph_count;
    uint32_t channel_count;
    uint32_t core_count;
    uint32_t reserved[4];
};

__aicore__ inline float FastExp(float value)
{
    if (value < -30.0F) return 0.0F;
    if (value > 0.0F) value = 0.0F;
    constexpr float inverseLn2 = 1.4426950408889634F;
    constexpr float ln2 = 0.6931471805599453F;
    const float rounded = value * inverseLn2 + (value >= 0.0F ? 0.5F : -0.5F);
    const int32_t exponent = static_cast<int32_t>(rounded);
    const float fraction = value - static_cast<float>(exponent) * ln2;
    const float fraction2 = fraction * fraction;
    const float polynomial = 1.0F + fraction + 0.5F * fraction2 + 0.1666666667F * fraction * fraction2 +
                             0.0416666667F * fraction2 * fraction2 + 0.0083333333F * fraction * fraction2 * fraction2 +
                             0.0013888889F * fraction2 * fraction2 * fraction2;
    int32_t bits = (exponent + 127) << 23;
    const float power2 = *reinterpret_cast<float*>(&bits);
    return polynomial * power2;
}

class AttentionalAggregationFusedKernel
{
   public:
    __aicore__ inline AttentionalAggregationFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR graphPtr, GM_ADDR features, GM_ADDR gates, GM_ADDR output,
                                const __gm__ AttentionKernelTiling* tiling)
    {
        nodes_ = tiling->node_count;
        graphs_ = tiling->graph_count;
        channels_ = tiling->channel_count;
        channelStride_ = (channels_ + 7U) / 8U * 8U;
        graphPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(graphPtr), graphs_ + 1U);
        featuresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(features),
                                    static_cast<uint64_t>(nodes_) * channels_);
        gatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gates), nodes_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), static_cast<uint64_t>(graphs_) * channels_);
        pipe_.InitBuffer(accumulatorBuf_, channelStride_ * sizeof(float));
        pipe_.InitBuffer(featureBuf_, channelStride_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> accumulator = accumulatorBuf_.Get<float>();
        LocalTensor<float> feature = featureBuf_.Get<float>();
        const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(channels_ * sizeof(float)), 0U, 0U, 0U};
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        for (uint32_t graph = core; graph < graphs_; graph += cores)
        {
            const int32_t begin = graphPtrGm_.GetValue(graph);
            const int32_t end = graphPtrGm_.GetValue(graph + 1U);
            float maximum = -3.402823e38F;
            for (int32_t node = begin; node < end; ++node)
            {
                const float gate = gatesGm_.GetValue(node);
                maximum = gate > maximum ? gate : maximum;
            }
            float denominator = 0.0F;
            for (int32_t node = begin; node < end; ++node)
            {
                denominator += FastExp(gatesGm_.GetValue(node) - maximum);
            }
            Duplicate(accumulator, 0.0F, channelStride_);
            pipe_barrier(PIPE_ALL);
            for (int32_t node = begin; node < end; ++node)
            {
                const float coefficient = FastExp(gatesGm_.GetValue(node) - maximum) / denominator;
                DataCopyPad(feature, featuresGm_[static_cast<uint64_t>(node) * channels_], copy, pad);
                pipe_barrier(PIPE_ALL);
                Muls(feature, feature, coefficient, channelStride_);
                pipe_barrier(PIPE_ALL);
                Add(accumulator, accumulator, feature, channelStride_);
                pipe_barrier(PIPE_ALL);
            }
            DataCopyPad(outputGm_[static_cast<uint64_t>(graph) * channels_], accumulator, copy);
            pipe_barrier(PIPE_ALL);
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> accumulatorBuf_, featureBuf_;
    GlobalTensor<int32_t> graphPtrGm_;
    GlobalTensor<float> featuresGm_, gatesGm_, outputGm_;
    uint32_t nodes_ = 0U;
    uint32_t graphs_ = 0U;
    uint32_t channels_ = 0U;
    uint32_t channelStride_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void attentional_aggregation_fused_kernel(GM_ADDR graphPtr, GM_ADDR features,
                                                                           GM_ADDR gates, GM_ADDR output,
                                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    AttentionalAggregationFusedKernel op;
    op.Init(graphPtr, features, gates, output, reinterpret_cast<const __gm__ AttentionKernelTiling*>(tiling));
    op.Process();
}
