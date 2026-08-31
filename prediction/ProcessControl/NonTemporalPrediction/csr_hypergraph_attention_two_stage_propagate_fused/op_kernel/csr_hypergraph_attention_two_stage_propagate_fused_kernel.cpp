/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct CsrHypergraphAttentionTwoStagePropagateFusedTiling
{
    uint32_t nodes;
    uint32_t hyperedges;
    uint32_t incidences;
    uint32_t heads;
    uint32_t channels;
    uint32_t maxEdgeSize;
    uint32_t maxNodeDegree;
    float negativeSlope;
};

template <typename T>
__aicore__ inline float ToCompute(T value)
{
    return static_cast<float>(value);
}

template <>
__aicore__ inline float ToCompute<bfloat16_t>(bfloat16_t value)
{
    return ToFloat(value);
}

template <typename T>
class HyperedgeAttentionAggregateKernel
{
   public:
    __aicore__ inline HyperedgeAttentionAggregateKernel() = default;

    __aicore__ inline void Init(GM_ADDR edgeRowPtr, GM_ADDR nodeIndex, GM_ADDR edgeScale, GM_ADDR features,
                                GM_ADDR attentionLogits, GM_ADDR normalizedAttention, GM_ADDR edgeFeatures,
                                const __gm__ CsrHypergraphAttentionTwoStagePropagateFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        hyperedges_ = tiling->hyperedges;
        incidences_ = tiling->incidences;
        heads_ = tiling->heads;
        channels_ = tiling->channels;
        maxEdgeSize_ = tiling->maxEdgeSize;
        negativeSlope_ = tiling->negativeSlope;
        segmentStride_ = (maxEdgeSize_ + 7U) / 8U * 8U;
        outputValues_ = heads_ * channels_;
        outputStride_ = (outputValues_ + 7U) / 8U * 8U;
        edgeRowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(edgeRowPtr), hyperedges_ + 1U);
        nodeIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(nodeIndex), incidences_);
        edgeScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(edgeScale), hyperedges_);
        featuresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(features),
                                    static_cast<uint64_t>(nodes_) * outputValues_);
        attentionLogitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attentionLogits),
                                           static_cast<uint64_t>(incidences_) * heads_);
        normalizedAttentionGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normalizedAttention),
                                               static_cast<uint64_t>(incidences_) * heads_);
        edgeFeaturesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(edgeFeatures),
                                        static_cast<uint64_t>(hyperedges_) * outputValues_);
        pipe_.InitBuffer(weightBuf_, segmentStride_ * sizeof(float));
        pipe_.InitBuffer(outputBuf_, outputStride_ * sizeof(float));
        scalarToVectorEvent_ = pipe_.AllocEventID<HardEvent::S_V>();
        vectorToScalarEvent_ = pipe_.AllocEventID<HardEvent::V_S>();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> weights = weightBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        const DataCopyExtParams outputCopy{1U, static_cast<uint32_t>(outputValues_ * sizeof(float)), 0U, 0U, 0U};
        for (uint32_t edgeId = GetBlockIdx(); edgeId < hyperedges_; edgeId += GetBlockNum())
        {
            Duplicate(output, 0.0F, outputStride_);
            const int32_t begin = edgeRowPtrGm_.GetValue(edgeId);
            const int32_t end = edgeRowPtrGm_.GetValue(edgeId + 1U);
            const uint32_t count = ValidEdge(begin, end) ? static_cast<uint32_t>(end - begin) : 0U;
            const float edgeScale = ToCompute(edgeScaleGm_.GetValue(edgeId));
            for (uint32_t head = 0U; head < heads_ && count > 0U; ++head)
            {
                Normalize(begin, count, head, weights);
                AccumulateEdge(begin, count, head, edgeScale, weights, output);
            }
            pipe_barrier(PIPE_ALL);
            DataCopyPad(edgeFeaturesGm_[static_cast<uint64_t>(edgeId) * outputValues_], output, outputCopy);
            pipe_barrier(PIPE_ALL);
        }
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent_);
        pipe_.ReleaseEventID<HardEvent::V_S>(vectorToScalarEvent_);
    }

   private:
    __aicore__ inline bool ValidEdge(int32_t begin, int32_t end)
    {
        bool valid = begin >= 0 && end >= begin && static_cast<uint32_t>(end) <= incidences_ &&
                     static_cast<uint32_t>(end - begin) <= maxEdgeSize_;
        for (int32_t incidence = begin; valid && incidence < end; ++incidence)
        {
            const int32_t node = nodeIndexGm_.GetValue(incidence);
            valid = node >= 0 && static_cast<uint32_t>(node) < nodes_;
        }
        return valid;
    }

    __aicore__ inline void Normalize(int32_t begin, uint32_t count, uint32_t head, LocalTensor<float>& weights)
    {
        float maximum = -3.402823466e38F;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t incidence = static_cast<uint32_t>(begin) + item;
            float score = ToCompute(attentionLogitsGm_.GetValue(static_cast<uint64_t>(incidence) * heads_ + head));
            score = score >= 0.0F ? score : score * negativeSlope_;
            weights.SetValue(item, score);
            maximum = score > maximum ? score : maximum;
        }
        SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
        WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);
        Adds(weights, weights, -maximum, count);
        Exp(weights, weights, count);
        SetFlag<HardEvent::V_S>(vectorToScalarEvent_);
        WaitFlag<HardEvent::V_S>(vectorToScalarEvent_);
        float denominator = 0.0F;
        for (uint32_t item = 0U; item < count; ++item) denominator += weights.GetValue(item);
        SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
        WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);
        Muls(weights, weights, denominator > 0.0F ? 1.0F / denominator : 0.0F, count);
        pipe_barrier(PIPE_ALL);
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(normalizedAttentionGm_[static_cast<uint64_t>(head) * incidences_ + begin], weights, copy);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void AccumulateEdge(int32_t begin, uint32_t count, uint32_t head, float scale,
                                          LocalTensor<float>& weights, LocalTensor<float>& output)
    {
        const uint32_t outputBase = head * channels_;
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t incidence = static_cast<uint32_t>(begin) + item;
            const uint32_t node = static_cast<uint32_t>(nodeIndexGm_.GetValue(incidence));
            const uint64_t featureBase = (static_cast<uint64_t>(node) * heads_ + head) * channels_;
            for (uint32_t channel = 0U; channel < channels_; ++channel)
            {
                const uint32_t offset = outputBase + channel;
                const float value =
                    scale * weights.GetValue(item) * ToCompute(featuresGm_.GetValue(featureBase + channel));
                output.SetValue(offset, output.GetValue(offset) + value);
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> weightBuf_, outputBuf_;
    GlobalTensor<int32_t> edgeRowPtrGm_, nodeIndexGm_;
    GlobalTensor<T> edgeScaleGm_, featuresGm_, attentionLogitsGm_;
    GlobalTensor<float> normalizedAttentionGm_, edgeFeaturesGm_;
    uint32_t nodes_ = 0U, hyperedges_ = 0U, incidences_ = 0U;
    uint32_t heads_ = 0U, channels_ = 0U, maxEdgeSize_ = 0U;
    uint32_t segmentStride_ = 0U, outputValues_ = 0U, outputStride_ = 0U;
    float negativeSlope_ = 0.2F;
    TEventID scalarToVectorEvent_ = 0U, vectorToScalarEvent_ = 0U;
};

template <typename T>
class NodeAttentionAggregateKernel
{
   public:
    __aicore__ inline NodeAttentionAggregateKernel() = default;

    __aicore__ inline void Init(GM_ADDR nodeRowPtr, GM_ADDR edgeIndex, GM_ADDR incidencePosition, GM_ADDR nodeScale,
                                GM_ADDR normalizedAttention, GM_ADDR edgeFeatures, GM_ADDR output,
                                const __gm__ CsrHypergraphAttentionTwoStagePropagateFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        hyperedges_ = tiling->hyperedges;
        incidences_ = tiling->incidences;
        heads_ = tiling->heads;
        channels_ = tiling->channels;
        maxNodeDegree_ = tiling->maxNodeDegree;
        outputValues_ = heads_ * channels_;
        outputStride_ = (outputValues_ + 7U) / 8U * 8U;
        nodeRowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(nodeRowPtr), nodes_ + 1U);
        edgeIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(edgeIndex), incidences_);
        incidencePositionGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(incidencePosition), incidences_);
        nodeScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(nodeScale), nodes_);
        normalizedAttentionGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normalizedAttention),
                                               static_cast<uint64_t>(incidences_) * heads_);
        edgeFeaturesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(edgeFeatures),
                                        static_cast<uint64_t>(hyperedges_) * outputValues_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), static_cast<uint64_t>(nodes_) * outputValues_);
        pipe_.InitBuffer(outputBuf_, outputStride_ * sizeof(float));
        if constexpr (!std::is_same_v<T, float>)
        {
            pipe_.InitBuffer(outputTypeBuf_, outputStride_ * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> output = outputBuf_.Get<float>();
        for (uint32_t node = GetBlockIdx(); node < nodes_; node += GetBlockNum())
        {
            Duplicate(output, 0.0F, outputStride_);
            const int32_t begin = nodeRowPtrGm_.GetValue(node);
            const int32_t end = nodeRowPtrGm_.GetValue(node + 1U);
            const uint32_t count = ValidNode(begin, end) ? static_cast<uint32_t>(end - begin) : 0U;
            const float scale = ToCompute(nodeScaleGm_.GetValue(node));
            AccumulateNode(begin, count, scale, output);
            pipe_barrier(PIPE_ALL);
            StoreNode(node, output);
        }
    }

   private:
    __aicore__ inline bool ValidNode(int32_t begin, int32_t end)
    {
        bool valid = begin >= 0 && end >= begin && static_cast<uint32_t>(end) <= incidences_ &&
                     static_cast<uint32_t>(end - begin) <= maxNodeDegree_;
        for (int32_t incidence = begin; valid && incidence < end; ++incidence)
        {
            const int32_t edge = edgeIndexGm_.GetValue(incidence);
            const int32_t position = incidencePositionGm_.GetValue(incidence);
            valid = edge >= 0 && static_cast<uint32_t>(edge) < hyperedges_ && position >= 0 &&
                    static_cast<uint32_t>(position) < incidences_;
        }
        return valid;
    }

    __aicore__ inline void AccumulateNode(int32_t begin, uint32_t count, float scale, LocalTensor<float>& output)
    {
        for (uint32_t item = 0U; item < count; ++item)
        {
            const uint32_t incidence = static_cast<uint32_t>(begin) + item;
            const uint32_t edge = static_cast<uint32_t>(edgeIndexGm_.GetValue(incidence));
            const uint32_t position = static_cast<uint32_t>(incidencePositionGm_.GetValue(incidence));
            for (uint32_t head = 0U; head < heads_; ++head)
            {
                const float alpha =
                    normalizedAttentionGm_.GetValue(static_cast<uint64_t>(head) * incidences_ + position);
                const uint64_t edgeBase = (static_cast<uint64_t>(edge) * heads_ + head) * channels_;
                const uint32_t outputBase = head * channels_;
                for (uint32_t channel = 0U; channel < channels_; ++channel)
                {
                    const uint32_t offset = outputBase + channel;
                    output.SetValue(
                        offset, output.GetValue(offset) + scale * alpha * edgeFeaturesGm_.GetValue(edgeBase + channel));
                }
            }
        }
    }

    __aicore__ inline void StoreNode(uint32_t node, LocalTensor<float>& output)
    {
        const uint64_t outputBase = static_cast<uint64_t>(node) * outputValues_;
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(outputValues_ * sizeof(T)), 0U, 0U, 0U};
        if constexpr (std::is_same_v<T, float>)
        {
            DataCopyPad(outputGm_[outputBase], output, copy);
        }
        else
        {
            LocalTensor<T> outputTyped = outputTypeBuf_.Get<T>();
            Cast(outputTyped, output, RoundMode::CAST_RINT, outputValues_);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(outputGm_[outputBase], outputTyped, copy);
        }
        pipe_barrier(PIPE_ALL);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> outputBuf_, outputTypeBuf_;
    GlobalTensor<int32_t> nodeRowPtrGm_, edgeIndexGm_, incidencePositionGm_;
    GlobalTensor<T> nodeScaleGm_, outputGm_;
    GlobalTensor<float> normalizedAttentionGm_, edgeFeaturesGm_;
    uint32_t nodes_ = 0U, hyperedges_ = 0U, incidences_ = 0U;
    uint32_t heads_ = 0U, channels_ = 0U, maxNodeDegree_ = 0U;
    uint32_t outputValues_ = 0U, outputStride_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_hypergraph_attention_node_to_edge_kernel_fp32(
    GM_ADDR edgeRowPtr, GM_ADDR nodeIndex, GM_ADDR edgeScale, GM_ADDR features, GM_ADDR attentionLogits,
    GM_ADDR normalizedAttention, GM_ADDR edgeFeatures, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    HyperedgeAttentionAggregateKernel<float> op;
    op.Init(edgeRowPtr, nodeIndex, edgeScale, features, attentionLogits, normalizedAttention, edgeFeatures,
            reinterpret_cast<const __gm__ CsrHypergraphAttentionTwoStagePropagateFusedTiling*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void csr_hypergraph_attention_edge_to_node_kernel_fp32(
    GM_ADDR nodeRowPtr, GM_ADDR edgeIndex, GM_ADDR incidencePosition, GM_ADDR nodeScale, GM_ADDR normalizedAttention,
    GM_ADDR edgeFeatures, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    NodeAttentionAggregateKernel<float> op;
    op.Init(nodeRowPtr, edgeIndex, incidencePosition, nodeScale, normalizedAttention, edgeFeatures, output,
            reinterpret_cast<const __gm__ CsrHypergraphAttentionTwoStagePropagateFusedTiling*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void csr_hypergraph_attention_node_to_edge_kernel_fp16(
    GM_ADDR edgeRowPtr, GM_ADDR nodeIndex, GM_ADDR edgeScale, GM_ADDR features, GM_ADDR attentionLogits,
    GM_ADDR normalizedAttention, GM_ADDR edgeFeatures, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    HyperedgeAttentionAggregateKernel<half> op;
    op.Init(edgeRowPtr, nodeIndex, edgeScale, features, attentionLogits, normalizedAttention, edgeFeatures,
            reinterpret_cast<const __gm__ CsrHypergraphAttentionTwoStagePropagateFusedTiling*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void csr_hypergraph_attention_edge_to_node_kernel_fp16(
    GM_ADDR nodeRowPtr, GM_ADDR edgeIndex, GM_ADDR incidencePosition, GM_ADDR nodeScale, GM_ADDR normalizedAttention,
    GM_ADDR edgeFeatures, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    NodeAttentionAggregateKernel<half> op;
    op.Init(nodeRowPtr, edgeIndex, incidencePosition, nodeScale, normalizedAttention, edgeFeatures, output,
            reinterpret_cast<const __gm__ CsrHypergraphAttentionTwoStagePropagateFusedTiling*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void csr_hypergraph_attention_node_to_edge_kernel_bf16(
    GM_ADDR edgeRowPtr, GM_ADDR nodeIndex, GM_ADDR edgeScale, GM_ADDR features, GM_ADDR attentionLogits,
    GM_ADDR normalizedAttention, GM_ADDR edgeFeatures, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    HyperedgeAttentionAggregateKernel<bfloat16_t> op;
    op.Init(edgeRowPtr, nodeIndex, edgeScale, features, attentionLogits, normalizedAttention, edgeFeatures,
            reinterpret_cast<const __gm__ CsrHypergraphAttentionTwoStagePropagateFusedTiling*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void csr_hypergraph_attention_edge_to_node_kernel_bf16(
    GM_ADDR nodeRowPtr, GM_ADDR edgeIndex, GM_ADDR incidencePosition, GM_ADDR nodeScale, GM_ADDR normalizedAttention,
    GM_ADDR edgeFeatures, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    NodeAttentionAggregateKernel<bfloat16_t> op;
    op.Init(nodeRowPtr, edgeIndex, incidencePosition, nodeScale, normalizedAttention, edgeFeatures, output,
            reinterpret_cast<const __gm__ CsrHypergraphAttentionTwoStagePropagateFusedTiling*>(tiling));
    op.Process();
}
