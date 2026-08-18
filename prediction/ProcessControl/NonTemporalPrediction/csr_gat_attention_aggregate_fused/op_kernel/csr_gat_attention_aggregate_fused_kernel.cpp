/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "../../attention_kernel_common.h"

using namespace AscendC;

namespace
{
class CsrGatAttentionAggregateFusedKernel
{
   public:
    __aicore__ inline CsrGatAttentionAggregateFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR projected, GM_ADDR attentionSource,
                                GM_ADDR attentionTarget, GM_ADDR output, const __gm__ AttentionFusedTiling* tiling)
    {
        shape_ = AttentionKernel::ReadShape(tiling);
        negativeSlope_ = tiling->parameter;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), shape_.nodes + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), shape_.edges);
        const uint64_t tensorSize = static_cast<uint64_t>(shape_.nodes) * shape_.outputValues;
        projectedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(projected), tensorSize);
        attentionSourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attentionSource), shape_.outputValues);
        attentionTargetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attentionTarget), shape_.outputValues);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), tensorSize);
        pipe_.InitBuffer(attentionBuf_, shape_.segmentStride * sizeof(float));
        pipe_.InitBuffer(outputBuf_, shape_.outputStride * sizeof(float));
        scalarToVectorEvent_ = pipe_.AllocEventID<HardEvent::S_V>();
        vectorToScalarEvent_ = pipe_.AllocEventID<HardEvent::V_S>();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> attention = attentionBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(shape_.outputValues * sizeof(float)), 0U, 0U, 0U};
        for (uint32_t target = GetBlockIdx(); target < shape_.nodes; target += GetBlockNum())
        {
            ProcessTarget(target, attention, output, copy);
        }
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent_);
        pipe_.ReleaseEventID<HardEvent::V_S>(vectorToScalarEvent_);
    }

   private:
    __aicore__ inline void ProcessTarget(uint32_t target, LocalTensor<float>& attention, LocalTensor<float>& output,
                                         const DataCopyExtParams& copy)
    {
        Duplicate(output, 0.0F, shape_.outputStride);
        int32_t begin = 0;
        const uint32_t count = AttentionKernel::SegmentSize(rowPtrGm_, sourceIndexGm_, target, shape_, &begin);
        for (uint32_t head = 0U; head < shape_.heads && count > 0U; ++head)
        {
            const float maximum = ComputeLogits(target, begin, count, head, attention);
            const float inverse =
                AttentionKernel::Normalize(attention, count, maximum, scalarToVectorEvent_, vectorToScalarEvent_);
            AttentionKernel::Accumulate(output, attention, sourceIndexGm_, projectedGm_, begin, count, head, shape_,
                                        inverse);
        }
        pipe_barrier(PIPE_ALL);
        DataCopyPad(outputGm_[static_cast<uint64_t>(target) * shape_.outputValues], output, copy);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline float TargetScore(uint32_t target, uint32_t head)
    {
        const uint64_t targetBase = (static_cast<uint64_t>(target) * shape_.heads + head) * shape_.channels;
        const uint64_t attentionBase = static_cast<uint64_t>(head) * shape_.channels;
        float score = 0.0F;
        for (uint32_t channel = 0U; channel < shape_.channels; ++channel)
        {
            score += projectedGm_.GetValue(targetBase + channel) * attentionTargetGm_.GetValue(attentionBase + channel);
        }
        return score;
    }

    __aicore__ inline float ComputeLogits(uint32_t target, int32_t begin, uint32_t count, uint32_t head,
                                          LocalTensor<float>& attention)
    {
        const uint64_t attentionBase = static_cast<uint64_t>(head) * shape_.channels;
        const float targetScore = TargetScore(target, head);
        float maximum = -3.402823466e38F;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(begin + item));
            const uint64_t sourceBase = (static_cast<uint64_t>(source) * shape_.heads + head) * shape_.channels;
            float logit = targetScore;
            for (uint32_t channel = 0U; channel < shape_.channels; ++channel)
            {
                logit +=
                    projectedGm_.GetValue(sourceBase + channel) * attentionSourceGm_.GetValue(attentionBase + channel);
            }
            logit = logit >= 0.0F ? logit : logit * negativeSlope_;
            attention.SetValue(item, logit);
            maximum = logit > maximum ? logit : maximum;
        }
        return maximum;
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> attentionBuf_, outputBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<float> projectedGm_, attentionSourceGm_, attentionTargetGm_, outputGm_;
    AttentionKernel::Shape shape_{};
    float negativeSlope_ = 0.2F;
    TEventID scalarToVectorEvent_ = 0, vectorToScalarEvent_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_gat_attention_aggregate_fused_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                               GM_ADDR projected,
                                                                               GM_ADDR attentionSource,
                                                                               GM_ADDR attentionTarget, GM_ADDR output,
                                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CsrGatAttentionAggregateFusedKernel op;
    op.Init(rowPtr, sourceIndex, projected, attentionSource, attentionTarget, output,
            reinterpret_cast<const __gm__ AttentionFusedTiling*>(tiling));
    op.Process();
}
