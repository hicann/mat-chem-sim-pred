/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "../../attention_kernel_common.h"

using namespace AscendC;

namespace
{
class CsrTransformerDotAttentionAggregateFusedKernel
{
   public:
    __aicore__ inline CsrTransformerDotAttentionAggregateFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                GM_ADDR output, const __gm__ AttentionFusedTiling* tiling)
    {
        shape_ = AttentionKernel::ReadShape(tiling);
        inverseScale_ = tiling->parameter;
        const uint64_t elements = static_cast<uint64_t>(shape_.nodes) * shape_.outputValues;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), shape_.nodes + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), shape_.edges);
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(query), elements);
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(key), elements);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(value), elements);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), elements);
        pipe_.InitBuffer(probabilityBuf_, shape_.segmentStride * sizeof(float));
        pipe_.InitBuffer(resultBuf_, shape_.outputStride * sizeof(float));
        scalarVectorEvent_ = pipe_.AllocEventID<HardEvent::S_V>();
        vectorScalarEvent_ = pipe_.AllocEventID<HardEvent::V_S>();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> probabilities = probabilityBuf_.Get<float>();
        LocalTensor<float> result = resultBuf_.Get<float>();
        const DataCopyExtParams copyParameters{1U, static_cast<uint32_t>(shape_.outputValues * sizeof(float)), 0U, 0U,
                                               0U};
        uint32_t target = GetBlockIdx();
        while (target < shape_.nodes)
        {
            RunTarget(target, probabilities, result, copyParameters);
            target += GetBlockNum();
        }
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarVectorEvent_);
        pipe_.ReleaseEventID<HardEvent::V_S>(vectorScalarEvent_);
    }

   private:
    __aicore__ inline void RunTarget(uint32_t target, LocalTensor<float>& probabilities, LocalTensor<float>& result,
                                     const DataCopyExtParams& copyParameters)
    {
        Duplicate(result, 0.0F, shape_.outputStride);
        int32_t begin = 0;
        const uint32_t count = AttentionKernel::SegmentSize(rowPtrGm_, sourceIndexGm_, target, shape_, &begin);
        if (count != 0U)
        {
            CalculateHeads(target, begin, count, probabilities, result);
        }
        pipe_barrier(PIPE_ALL);
        DataCopyPad(outputGm_[static_cast<uint64_t>(target) * shape_.outputValues], result, copyParameters);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void CalculateHeads(uint32_t target, int32_t begin, uint32_t count,
                                          LocalTensor<float>& probabilities, LocalTensor<float>& result)
    {
        for (uint32_t head = 0U; head < shape_.heads; ++head)
        {
            const float maximum = DotLogits(target, begin, count, head, probabilities);
            const float inverse =
                AttentionKernel::Normalize(probabilities, count, maximum, scalarVectorEvent_, vectorScalarEvent_);
            AttentionKernel::Accumulate(result, probabilities, sourceIndexGm_, valueGm_, begin, count, head, shape_,
                                        inverse);
        }
    }

    __aicore__ inline float DotLogits(uint32_t target, int32_t begin, uint32_t count, uint32_t head,
                                      LocalTensor<float>& probabilities)
    {
        float maximum = -3.402823466e38F;
        const uint64_t queryBase = (static_cast<uint64_t>(target) * shape_.heads + head) * shape_.channels;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t source = static_cast<uint32_t>(sourceIndexGm_.GetValue(begin + item));
            const uint64_t keyBase = (static_cast<uint64_t>(source) * shape_.heads + head) * shape_.channels;
            float logit = 0.0F;
            for (uint32_t channel = 0U; channel < shape_.channels; ++channel)
            {
                logit += queryGm_.GetValue(queryBase + channel) * keyGm_.GetValue(keyBase + channel);
            }
            logit *= inverseScale_;
            probabilities.SetValue(item, logit);
            maximum = logit > maximum ? logit : maximum;
        }
        return maximum;
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> probabilityBuf_, resultBuf_;
    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<float> queryGm_, keyGm_, valueGm_, outputGm_;
    AttentionKernel::Shape shape_{};
    float inverseScale_ = 1.0F;
    TEventID scalarVectorEvent_ = 0, vectorScalarEvent_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_transformer_dot_attention_aggregate_fused_kernel(
    GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    CsrTransformerDotAttentionAggregateFusedKernel op;
    op.Init(rowPtr, sourceIndex, query, key, value, output,
            reinterpret_cast<const __gm__ AttentionFusedTiling*>(tiling));
    op.Process();
}
