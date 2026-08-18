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

struct FarthestPointTiling
{
    uint32_t batch;
    uint32_t pointCount;
    uint32_t sampleCount;
    uint32_t coreNum;
};

class FarthestPointSamplingFusedKernel
{
   public:
    __aicore__ inline FarthestPointSamplingFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR points, GM_ADDR sampleIndices, const __gm__ FarthestPointTiling* tiling)
    {
        batch_ = tiling->batch;
        pointCount_ = tiling->pointCount;
        sampleCount_ = tiling->sampleCount;
        vectorPoints_ = (pointCount_ + 63U) & ~63U;
        pointValues_ = pointCount_ * 3U;
        pointValuesStride_ = (pointValues_ + 7U) / 8U * 8U;
        pointsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(points),
                                  static_cast<uint64_t>(batch_) * pointValues_);
        sampleIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sampleIndices),
                                         static_cast<uint64_t>(batch_) * sampleCount_);
        pipe_.InitBuffer(pointsBuf_, pointValuesStride_ * sizeof(float));
        const uint32_t vectorBytes = vectorPoints_ * sizeof(float);
        pipe_.InitBuffer(xBuf_, vectorBytes);
        pipe_.InitBuffer(yBuf_, vectorBytes);
        pipe_.InitBuffer(zBuf_, vectorBytes);
        pipe_.InitBuffer(workBuf_, vectorBytes);
        pipe_.InitBuffer(distanceBuf_, vectorBytes);
        pipe_.InitBuffer(minimumDistanceBuf_, vectorBytes);
        scalarToVectorEvent_ = pipe_.AllocEventID<HardEvent::S_V>();
        vectorToScalarEvent_ = pipe_.AllocEventID<HardEvent::V_S>();
    }

    __aicore__ inline void LoadCoordinates(uint64_t batch)
    {
        LocalTensor<float> points = pointsBuf_.Get<float>();
        LocalTensor<float> x = xBuf_.Get<float>();
        LocalTensor<float> y = yBuf_.Get<float>();
        LocalTensor<float> z = zBuf_.Get<float>();
        LocalTensor<float> minimumDistance = minimumDistanceBuf_.Get<float>();
        const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        DataCopyExtParams pointCp{1U, static_cast<uint32_t>(pointValues_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(points, pointsGm_[batch * pointValues_], pointCp, pad);
        pipe_barrier(PIPE_ALL);
        for (uint32_t point = 0U; point < pointCount_; ++point)
        {
            x.SetValue(point, points.GetValue(point * 3U));
            y.SetValue(point, points.GetValue(point * 3U + 1U));
            z.SetValue(point, points.GetValue(point * 3U + 2U));
        }
        SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
        WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);
        Duplicate(minimumDistance, 3.0e30f, vectorPoints_);
    }

    __aicore__ inline void UpdateMinimumDistance(uint32_t selected)
    {
        LocalTensor<float> x = xBuf_.Get<float>();
        LocalTensor<float> y = yBuf_.Get<float>();
        LocalTensor<float> z = zBuf_.Get<float>();
        LocalTensor<float> work = workBuf_.Get<float>();
        LocalTensor<float> distance = distanceBuf_.Get<float>();
        LocalTensor<float> minimumDistance = minimumDistanceBuf_.Get<float>();
        const float sourceX = x.GetValue(selected);
        const float sourceY = y.GetValue(selected);
        const float sourceZ = z.GetValue(selected);
        SetFlag<HardEvent::S_V>(scalarToVectorEvent_);
        WaitFlag<HardEvent::S_V>(scalarToVectorEvent_);
        Adds(work, x, -sourceX, vectorPoints_);
        Mul(distance, work, work, vectorPoints_);
        Adds(work, y, -sourceY, vectorPoints_);
        Mul(work, work, work, vectorPoints_);
        Add(distance, distance, work, vectorPoints_);
        Adds(work, z, -sourceZ, vectorPoints_);
        Mul(work, work, work, vectorPoints_);
        Add(distance, distance, work, vectorPoints_);
        Min(minimumDistance, minimumDistance, distance, vectorPoints_);
        SetFlag<HardEvent::V_S>(vectorToScalarEvent_);
        WaitFlag<HardEvent::V_S>(vectorToScalarEvent_);
    }

    __aicore__ inline uint32_t FindFarthest()
    {
        LocalTensor<float> minimumDistance = minimumDistanceBuf_.Get<float>();
        float farthestDistance = -1.0f;
        uint32_t farthestIndex = 0U;
        for (uint32_t point = 0U; point < pointCount_; ++point)
        {
            const float candidate = minimumDistance.GetValue(point);
            if (candidate > farthestDistance)
            {
                farthestDistance = candidate;
                farthestIndex = point;
            }
        }
        return farthestIndex;
    }

    __aicore__ inline void ProcessBatch(uint64_t batch)
    {
        LoadCoordinates(batch);
        uint32_t selected = 0U;
        for (uint32_t sample = 0U; sample < sampleCount_; ++sample)
        {
            sampleIndicesGm_.SetValue(static_cast<uint64_t>(batch) * sampleCount_ + sample,
                                      static_cast<int32_t>(selected));
            if (sample + 1U == sampleCount_)
            {
                break;
            }
            UpdateMinimumDistance(selected);
            selected = FindFarthest();
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        // CWE-190 fix: a 64-bit index prevents wraparound near the UINT32 batch limit.
        for (uint64_t batch = core; batch < batch_; batch += cores)
        {
            ProcessBatch(batch);
        }
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent_);
        pipe_.ReleaseEventID<HardEvent::V_S>(vectorToScalarEvent_);
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> pointsBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> zBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> distanceBuf_;
    TBuf<TPosition::VECCALC> minimumDistanceBuf_;
    GlobalTensor<float> pointsGm_;
    GlobalTensor<int32_t> sampleIndicesGm_;
    uint32_t batch_ = 0U;
    uint32_t pointCount_ = 0U;
    uint32_t sampleCount_ = 0U;
    uint32_t vectorPoints_ = 0U;
    uint32_t pointValues_ = 0U;
    uint32_t pointValuesStride_ = 0U;
    TEventID scalarToVectorEvent_ = 0;
    TEventID vectorToScalarEvent_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void farthest_point_sampling_fused_kernel(GM_ADDR points, GM_ADDR sampleIndices,
                                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    const __gm__ FarthestPointTiling* params = reinterpret_cast<const __gm__ FarthestPointTiling*>(tiling);
    FarthestPointSamplingFusedKernel op;
    op.Init(points, sampleIndices, params);
    op.Process();
}
