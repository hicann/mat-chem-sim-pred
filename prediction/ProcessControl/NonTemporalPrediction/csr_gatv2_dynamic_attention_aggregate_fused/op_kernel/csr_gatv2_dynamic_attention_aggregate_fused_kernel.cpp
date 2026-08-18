/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "../../attention_kernel_common.h"

using namespace AscendC;

namespace
{
class CsrGatv2DynamicAttentionAggregateFusedKernel
{
   public:
    __aicore__ inline CsrGatv2DynamicAttentionAggregateFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR sourceProjected, GM_ADDR targetProjected,
                                GM_ADDR attention, GM_ADDR output, const __gm__ AttentionFusedTiling* tiling)
    {
        shape_ = AttentionKernel::ReadShape(tiling);
        slope_ = tiling->parameter;
        const uint64_t projectionCount = static_cast<uint64_t>(shape_.nodes) * shape_.outputValues;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), shape_.nodes + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), shape_.edges);
        sourceProjectedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sourceProjected), projectionCount);
        targetProjectedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(targetProjected), projectionCount);
        attentionGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attention), shape_.outputValues);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), projectionCount);
        pipe_.InitBuffer(weightBuf_, shape_.segmentStride * sizeof(float));
        pipe_.InitBuffer(outputBuf_, shape_.outputStride * sizeof(float));
        toVector_ = pipe_.AllocEventID<HardEvent::S_V>();
        toScalar_ = pipe_.AllocEventID<HardEvent::V_S>();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> weights = weightBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        for (uint32_t target = GetBlockIdx(); target < shape_.nodes; target += GetBlockNum())
        {
            ProcessOneTarget(target, weights, output);
        }
        pipe_.ReleaseEventID<HardEvent::S_V>(toVector_);
        pipe_.ReleaseEventID<HardEvent::V_S>(toScalar_);
    }

   private:
    __aicore__ inline void ProcessOneTarget(uint32_t target, LocalTensor<float>& weights, LocalTensor<float>& output)
    {
        Duplicate(output, 0.0F, shape_.outputStride);
        int32_t firstEdge = 0;
        const uint32_t edgeCount = AttentionKernel::SegmentSize(rowPtrGm_, sourceIndexGm_, target, shape_, &firstEdge);
        ProcessHeads(target, firstEdge, edgeCount, weights, output);
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(shape_.outputValues * sizeof(float)), 0U, 0U, 0U};
        pipe_barrier(PIPE_ALL);
        DataCopyPad(outputGm_[static_cast<uint64_t>(target) * shape_.outputValues], output, copy);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ProcessHeads(uint32_t target, int32_t begin, uint32_t count, LocalTensor<float>& weights,
                                        LocalTensor<float>& output)
    {
        for (uint32_t head = 0U; head < shape_.heads && count != 0U; ++head)
        {
            const float maximum = FillWeights(target, begin, count, head, weights);
            const float inverse = AttentionKernel::Normalize(weights, count, maximum, toVector_, toScalar_);
            AttentionKernel::Accumulate(output, weights, sourceIndexGm_, sourceProjectedGm_, begin, count, head, shape_,
                                        inverse);
        }
    }

    __aicore__ inline float FillWeights(uint32_t target, int32_t begin, uint32_t count, uint32_t head,
                                        LocalTensor<float>& weights)
    {
        float maximum = -3.402823466e38F;
        const uint64_t targetBase = (static_cast<uint64_t>(target) * shape_.heads + head) * shape_.channels;
        const uint64_t attentionBase = static_cast<uint64_t>(head) * shape_.channels;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(begin + item));
            const uint64_t sourceBase = (static_cast<uint64_t>(source) * shape_.heads + head) * shape_.channels;
            float logit = 0.0F;
            for (uint32_t channel = 0U; channel < shape_.channels; ++channel)
            {
                float value = sourceProjectedGm_.GetValue(sourceBase + channel) +
                              targetProjectedGm_.GetValue(targetBase + channel);
                value = value >= 0.0F ? value : value * slope_;
                logit += value * attentionGm_.GetValue(attentionBase + channel);
            }
            weights.SetValue(item, logit);
            maximum = logit > maximum ? logit : maximum;
        }
        return maximum;
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> weightBuf_, outputBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<float> sourceProjectedGm_, targetProjectedGm_, attentionGm_, outputGm_;
    AttentionKernel::Shape shape_{};
    float slope_ = 0.2F;
    TEventID toVector_ = 0, toScalar_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_gatv2_dynamic_attention_aggregate_fused_kernel(
    GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR sourceProjected, GM_ADDR targetProjected, GM_ADDR attention,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrGatv2DynamicAttentionAggregateFusedKernel op;
    op.Init(rowPtr, sourceIndex, sourceProjected, targetProjected, attention, output,
            reinterpret_cast<const __gm__ AttentionFusedTiling*>(tiling));
    op.Process();
}
