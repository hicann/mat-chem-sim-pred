/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
constexpr float kTiny = 1.0e-20F;
constexpr float kPi = 3.14159265358979323846F;
constexpr float kHalfPi = 1.57079632679489661923F;
constexpr float kQuarterPi = 0.78539816339744830962F;
constexpr float kTanPiOverEight = 0.41421356237309504880F;

struct DotProducts
{
    float first;
    float second;
    float third;
};

class PpfPointPairFeaturesFusedKernel
{
   public:
    __aicore__ inline PpfPointPairFeaturesFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR position, GM_ADDR normal, GM_ADDR sourceIndex, GM_ADDR targetIndex,
                                GM_ADDR output, uint32_t nodes, uint32_t edges)
    {
        nodes_ = nodes;
        edges_ = edges;
        positionGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(position), static_cast<uint64_t>(nodes_) * 3U);
        normalGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normal), static_cast<uint64_t>(nodes_) * 3U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        targetIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(targetIndex), edges_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), static_cast<uint64_t>(edges_) * 4U);
        pipe_.InitBuffer(rootInputBuf_, 8U * sizeof(float));
        pipe_.InitBuffer(rootOutputBuf_, 8U * sizeof(float));
        pipe_.InitBuffer(outputBuf_, 8U * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> rootInput = rootInputBuf_.Get<float>();
        LocalTensor<float> rootOutput = rootOutputBuf_.Get<float>();
        LocalTensor<float> edgeOutput = outputBuf_.Get<float>();
        const DataCopyExtParams outputCopy{1U, 4U * sizeof(float), 0U, 0U, 0U};

        for (uint32_t edge = GetBlockIdx(); edge < edges_; edge += GetBlockNum())
        {
            ProcessEdge(edge, rootInput, rootOutput, edgeOutput, outputCopy);
        }
    }

   private:
    __aicore__ inline DotProducts FillRootInputs(uint64_t sourceBase, uint64_t targetBase,
                                                 LocalTensor<float> rootInput)
    {
        const float px = positionGm_.GetValue(sourceBase) - positionGm_.GetValue(targetBase);
        const float py = positionGm_.GetValue(sourceBase + 1U) - positionGm_.GetValue(targetBase + 1U);
        const float pz = positionGm_.GetValue(sourceBase + 2U) - positionGm_.GetValue(targetBase + 2U);
        const float nix = normalGm_.GetValue(targetBase);
        const float niy = normalGm_.GetValue(targetBase + 1U);
        const float niz = normalGm_.GetValue(targetBase + 2U);
        const float njx = normalGm_.GetValue(sourceBase);
        const float njy = normalGm_.GetValue(sourceBase + 1U);
        const float njz = normalGm_.GetValue(sourceBase + 2U);
        const float c0x = niy * pz - niz * py;
        const float c0y = niz * px - nix * pz;
        const float c0z = nix * py - niy * px;
        const float c1x = njy * pz - njz * py;
        const float c1y = njz * px - njx * pz;
        const float c1z = njx * py - njy * px;
        const float c2x = niy * njz - niz * njy;
        const float c2y = niz * njx - nix * njz;
        const float c2z = nix * njy - niy * njx;
        Duplicate(rootInput, 0.0F, 8U);
        rootInput.SetValue(0U, px * px + py * py + pz * pz);
        rootInput.SetValue(1U, c0x * c0x + c0y * c0y + c0z * c0z);
        rootInput.SetValue(2U, c1x * c1x + c1y * c1y + c1z * c1z);
        rootInput.SetValue(3U, c2x * c2x + c2y * c2y + c2z * c2z);
        return {nix * px + niy * py + niz * pz, njx * px + njy * py + njz * pz,
                nix * njx + niy * njy + niz * njz};
    }

    __aicore__ inline void ProcessEdge(uint32_t edge, LocalTensor<float> rootInput, LocalTensor<float> rootOutput,
                                       LocalTensor<float> edgeOutput, const DataCopyExtParams& outputCopy)
    {
        const int32_t source = sourceIndexGm_.GetValue(edge);
        const int32_t target = targetIndexGm_.GetValue(edge);
        if (source < 0 || target < 0 || static_cast<uint32_t>(source) >= nodes_ ||
            static_cast<uint32_t>(target) >= nodes_)
        {
            Duplicate(edgeOutput, 0.0F, 8U);
            DataCopyPad(outputGm_[static_cast<uint64_t>(edge) * 4U], edgeOutput, outputCopy);
            return;
        }
        const uint64_t sourceBase = static_cast<uint64_t>(source) * 3U;
        const uint64_t targetBase = static_cast<uint64_t>(target) * 3U;
        const DotProducts dots = FillRootInputs(sourceBase, targetBase, rootInput);
        pipe_barrier(PIPE_ALL);
        Sqrt(rootOutput, rootInput, 8U);
        pipe_barrier(PIPE_ALL);
        edgeOutput.SetValue(0U, rootOutput.GetValue(0U));
        edgeOutput.SetValue(1U, ApproximateAtan2(rootOutput.GetValue(1U), dots.first));
        edgeOutput.SetValue(2U, ApproximateAtan2(rootOutput.GetValue(2U), dots.second));
        edgeOutput.SetValue(3U, ApproximateAtan2(rootOutput.GetValue(3U), dots.third));
        pipe_barrier(PIPE_ALL);
        DataCopyPad(outputGm_[static_cast<uint64_t>(edge) * 4U], edgeOutput, outputCopy);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline float ApproximateAtan(float value)
    {
        float reduced = value;
        float base = 0.0F;
        if (value > kTanPiOverEight)
        {
            reduced = (value - 1.0F) / (value + 1.0F);
            base = kQuarterPi;
        }
        const float square = reduced * reduced;
        float polynomial = 0.0805374449538F;
        polynomial = polynomial * square - 0.138776856032F;
        polynomial = polynomial * square + 0.199777106478F;
        polynomial = polynomial * square - 0.333329491539F;
        return base + reduced + reduced * square * polynomial;
    }

    __aicore__ inline float ApproximateAtan2(float cross, float dot)
    {
        const float absoluteDot = dot < 0.0F ? -dot : dot;
        if (cross <= kTiny && absoluteDot <= kTiny)
        {
            return 0.0F;
        }
        if (cross <= kTiny)
        {
            return dot < 0.0F ? kPi : 0.0F;
        }
        float acute;
        if (cross <= absoluteDot)
        {
            acute = ApproximateAtan(cross / absoluteDot);
        }
        else
        {
            const float safeCross = cross > kTiny ? cross : kTiny;
            acute = kHalfPi - ApproximateAtan(absoluteDot / safeCross);
        }
        return dot < 0.0F ? kPi - acute : acute;
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rootInputBuf_, rootOutputBuf_, outputBuf_;
    GlobalTensor<float> positionGm_, normalGm_, outputGm_;
    GlobalTensor<int32_t> sourceIndexGm_, targetIndexGm_;
    uint32_t nodes_ = 0U;
    uint32_t edges_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void ppf_point_pair_features_fused_kernel(GM_ADDR position, GM_ADDR normal,
                                                                           GM_ADDR sourceIndex, GM_ADDR targetIndex,
                                                                           GM_ADDR output, uint32_t nodes,
                                                                           uint32_t edges)
{
    PpfPointPairFeaturesFusedKernel op;
    op.Init(position, normal, sourceIndex, targetIndex, output, nodes, edges);
    op.Process();
}
