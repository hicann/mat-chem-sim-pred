/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct Tiling
{
    uint32_t nodes, positiveEdges, negativeEdges, channels;
    uint32_t coreNum, maxSegmentSize, reserved0, reserved1;
};
class Kernel
{
   public:
    __aicore__ inline Kernel() = default;
    __aicore__ inline void Init(GM_ADDR positiveRowPtr, GM_ADDR positiveSourceIndex, GM_ADDR negativeRowPtr,
                                GM_ADDR negativeSourceIndex, GM_ADDR features, GM_ADDR positiveInverseDegree,
                                GM_ADDR negativeInverseDegree, GM_ADDR output, const __gm__ Tiling* tiling)
    {
        nodes_ = tiling->nodes;
        positiveEdges_ = tiling->positiveEdges;
        negativeEdges_ = tiling->negativeEdges;
        channels_ = tiling->channels;
        maxSegmentSize_ = tiling->maxSegmentSize;
        featureValues_ = 2U * channels_;
        outputValues_ = 4U * channels_;
        featureStride_ = (featureValues_ + 7U) / 8U * 8U;
        outputStride_ = (outputValues_ + 7U) / 8U * 8U;
        positiveRowGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(positiveRowPtr), nodes_ + 1U);
        negativeRowGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(negativeRowPtr), nodes_ + 1U);
        positiveSourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(positiveSourceIndex), positiveEdges_);
        negativeSourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(negativeSourceIndex), negativeEdges_);
        featuresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(features),
                                    static_cast<uint64_t>(nodes_) * 2U * channels_);
        positiveInverseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(positiveInverseDegree), nodes_);
        negativeInverseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(negativeInverseDegree), nodes_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output),
                                  static_cast<uint64_t>(nodes_) * outputValues_);
        pipe_.InitBuffer(featureBuf_, featureStride_ * sizeof(float));
        pipe_.InitBuffer(outputBuf_, outputStride_ * sizeof(float));
    }

    __aicore__ inline bool ValidateRow(int32_t begin, int32_t end, uint32_t edgeCount, GlobalTensor<int32_t>& sources)
    {
        bool valid = begin >= 0 && end >= begin && static_cast<uint32_t>(end) <= edgeCount &&
                     static_cast<uint32_t>(end - begin) <= maxSegmentSize_;
        for (int32_t edge = begin; valid && edge < end; ++edge)
        {
            int32_t source = sources.GetValue(edge);
            valid = source >= 0 && static_cast<uint32_t>(source) < nodes_;
        }
        return valid;
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> feature = featureBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
        const DataCopyExtParams featureCopy{1U, static_cast<uint32_t>(featureValues_ * sizeof(float)), 0U, 0U, 0U};
        const DataCopyExtParams outputCopy{1U, static_cast<uint32_t>(outputValues_ * sizeof(float)), 0U, 0U, 0U};
        for (uint32_t target = GetBlockIdx(); target < nodes_; target += GetBlockNum())
        {
            Duplicate(output, 0.0F, outputStride_);
            int32_t pBegin = positiveRowGm_.GetValue(target);
            int32_t pEnd = positiveRowGm_.GetValue(target + 1U);
            int32_t nBegin = negativeRowGm_.GetValue(target);
            int32_t nEnd = negativeRowGm_.GetValue(target + 1U);
            bool positiveValid = ValidateRow(pBegin, pEnd, positiveEdges_, positiveSourceGm_);
            bool negativeValid = ValidateRow(nBegin, nEnd, negativeEdges_, negativeSourceGm_);
            float positiveInverse = positiveValid && pEnd > pBegin ? positiveInverseGm_.GetValue(target) : 0.0F;
            float negativeInverse = negativeValid && nEnd > nBegin ? negativeInverseGm_.GetValue(target) : 0.0F;
            for (int32_t edge = pBegin; positiveValid && edge < pEnd; ++edge)
            {
                uint32_t source = static_cast<uint32_t>(positiveSourceGm_.GetValue(edge));
                uint64_t base = static_cast<uint64_t>(source) * 2U * channels_;
                DataCopyPad(feature, featuresGm_[base], featureCopy, pad);
                pipe_barrier(PIPE_ALL);
                Add(output, output, feature, channels_);
                Add(output[2U * channels_], output[2U * channels_], feature[channels_], channels_);
                pipe_barrier(PIPE_ALL);
            }
            for (int32_t edge = nBegin; negativeValid && edge < nEnd; ++edge)
            {
                uint32_t source = static_cast<uint32_t>(negativeSourceGm_.GetValue(edge));
                uint64_t base = static_cast<uint64_t>(source) * 2U * channels_;
                DataCopyPad(feature, featuresGm_[base], featureCopy, pad);
                pipe_barrier(PIPE_ALL);
                Add(output[channels_], output[channels_], feature[channels_], channels_);
                Add(output[3U * channels_], output[3U * channels_], feature, channels_);
                pipe_barrier(PIPE_ALL);
            }
            Muls(output, output, positiveInverse, channels_);
            Muls(output[2U * channels_], output[2U * channels_], positiveInverse, channels_);
            Muls(output[channels_], output[channels_], negativeInverse, channels_);
            Muls(output[3U * channels_], output[3U * channels_], negativeInverse, channels_);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(outputGm_[static_cast<uint64_t>(target) * outputValues_], output, outputCopy);
            pipe_barrier(PIPE_ALL);
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> featureBuf_, outputBuf_;
    GlobalTensor<int32_t> positiveRowGm_, negativeRowGm_;
    GlobalTensor<int32_t> positiveSourceGm_, negativeSourceGm_;
    GlobalTensor<float> featuresGm_, positiveInverseGm_, negativeInverseGm_;
    GlobalTensor<float> outputGm_;
    uint32_t nodes_ = 0U, positiveEdges_ = 0U, negativeEdges_ = 0U;
    uint32_t channels_ = 0U, maxSegmentSize_ = 0U;
    uint32_t featureValues_ = 0U, outputValues_ = 0U;
    uint32_t featureStride_ = 0U, outputStride_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void csr_signed_cross_mean_pack_fused_kernel(
    GM_ADDR positiveRowPtr, GM_ADDR positiveSourceIndex, GM_ADDR negativeRowPtr, GM_ADDR negativeSourceIndex,
    GM_ADDR features, GM_ADDR positiveInverseDegree, GM_ADDR negativeInverseDegree, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    Kernel op;
    op.Init(positiveRowPtr, positiveSourceIndex, negativeRowPtr, negativeSourceIndex, features, positiveInverseDegree,
            negativeInverseDegree, output, reinterpret_cast<const __gm__ Tiling*>(tiling));
    op.Process();
}
