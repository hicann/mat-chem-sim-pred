/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include "../point_ball_query_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace
{

class PointBallQueryFusedKernel
{
   public:
    __aicore__ inline PointBallQueryFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR points, GM_ADDR queries, GM_ADDR indices, GM_ADDR counts,
                                const __gm__ BallQueryTiling* tiling)
    {
        batch_ = tiling->batch;
        pointCount_ = tiling->pointCount;
        queryCount_ = tiling->queryCount;
        sampleCount_ = tiling->sampleCount;
        totalQueries_ = tiling->totalQueries;
        radiusSquared_ = tiling->radiusSquared;
        indexStride_ = (sampleCount_ + 7U) / 8U * 8U;
        pointsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(points),
                                  static_cast<uint64_t>(batch_) * pointCount_ * 3U);
        queriesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(queries), static_cast<uint64_t>(totalQueries_) * 3U);
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices),
                                   static_cast<uint64_t>(totalQueries_) * sampleCount_);
        countsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts), totalQueries_);
        pipe_.InitBuffer(indexBuf_, indexStride_ * sizeof(int32_t));
        pipe_.InitBuffer(countBuf_, 8U * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> indices = indexBuf_.Get<int32_t>();
        LocalTensor<int32_t> countOutput = countBuf_.Get<int32_t>();
        DataCopyExtParams indexCp{1U, static_cast<uint32_t>(sampleCount_ * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyExtParams scalarCp{1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        for (uint32_t query = core; query < totalQueries_; query += cores)
        {
            const uint32_t batch = query / queryCount_;
            const uint64_t queryOffset = static_cast<uint64_t>(query) * 3U;
            const float qx = queriesGm_.GetValue(queryOffset);
            const float qy = queriesGm_.GetValue(queryOffset + 1U);
            const float qz = queriesGm_.GetValue(queryOffset + 2U);
            for (uint32_t slot = 0U; slot < sampleCount_; ++slot)
            {
                indices.SetValue(slot, -1);
            }
            uint32_t found = 0U;
            const uint64_t pointOffset = static_cast<uint64_t>(batch) * pointCount_ * 3U;
            for (uint32_t point = 0U; point < pointCount_ && found < sampleCount_; ++point)
            {
                const uint64_t offset = pointOffset + static_cast<uint64_t>(point) * 3U;
                const float dx = pointsGm_.GetValue(offset) - qx;
                const float dy = pointsGm_.GetValue(offset + 1U) - qy;
                const float dz = pointsGm_.GetValue(offset + 2U) - qz;
                if (dx * dx + dy * dy + dz * dz <= radiusSquared_)
                {
                    indices.SetValue(found, static_cast<int32_t>(point));
                    ++found;
                }
            }
            countOutput.SetValue(0U, static_cast<int32_t>(found));
            pipe_barrier(PIPE_ALL);
            DataCopyPad(indicesGm_[static_cast<uint64_t>(query) * sampleCount_], indices, indexCp);
            DataCopyPad(countsGm_[query], countOutput, scalarCp);
            pipe_barrier(PIPE_ALL);
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> countBuf_;
    GlobalTensor<float> pointsGm_;
    GlobalTensor<float> queriesGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> countsGm_;
    uint32_t batch_ = 0U;
    uint32_t pointCount_ = 0U;
    uint32_t queryCount_ = 0U;
    uint32_t sampleCount_ = 0U;
    uint32_t totalQueries_ = 0U;
    uint32_t indexStride_ = 0U;
    float radiusSquared_ = 0.0f;
};

}  // namespace

extern "C" __global__ __aicore__ void point_ball_query_fused_kernel(GM_ADDR points, GM_ADDR queries, GM_ADDR indices,
                                                                    GM_ADDR counts, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    PointBallQueryFusedKernel op;
    op.Init(points, queries, indices, counts, reinterpret_cast<const __gm__ BallQueryTiling*>(tiling));
    op.Process();
}
