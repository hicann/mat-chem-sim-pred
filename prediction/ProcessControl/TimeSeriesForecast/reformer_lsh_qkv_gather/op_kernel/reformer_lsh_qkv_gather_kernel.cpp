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

struct LshGatherTiling
{
    uint32_t rows;
    uint32_t source_length;
    uint32_t index_length;
    uint32_t width;
};

class LshQkvGatherKernel
{
   public:
    __aicore__ inline LshQkvGatherKernel() = default;

    __aicore__ inline void Init(GM_ADDR queryKey, GM_ADDR value, GM_ADDR indices, GM_ADDR sortedQueryKey,
                                GM_ADDR sortedValue, const LshGatherTiling& tiling)
    {
        rows_ = tiling.rows;
        sourceLength_ = tiling.source_length;
        indexLength_ = tiling.index_length;
        width_ = tiling.width;
        const uint64_t sourceElements = static_cast<uint64_t>(rows_) * sourceLength_ * width_;
        const uint64_t outputElements = static_cast<uint64_t>(rows_) * indexLength_ * width_;
        const uint64_t indexElements = static_cast<uint64_t>(rows_) * indexLength_;
        queryKeyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(queryKey), sourceElements);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(value), sourceElements);
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices), indexElements);
        sortedQueryKeyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sortedQueryKey), outputElements);
        sortedValueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sortedValue), outputElements);
        pipe_.InitBuffer(queryKeyBuffer_, width_ * sizeof(float));
        pipe_.InitBuffer(valueBuffer_, width_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = GetBlockIdx();
        const uint32_t blocks = GetBlockNum();
        const uint64_t jobs = static_cast<uint64_t>(rows_) * indexLength_;
        LocalTensor<float> queryKey = queryKeyBuffer_.Get<float>();
        LocalTensor<float> value = valueBuffer_.Get<float>();
        for (uint64_t job = block; job < jobs; job += blocks)
        {
            const uint32_t row = static_cast<uint32_t>(job / indexLength_);
            const int64_t sourceIndex = indicesGm_.GetValue(job);
            const uint64_t source = (static_cast<uint64_t>(row) * sourceLength_ + sourceIndex) * width_;
            const uint64_t output = job * width_;
            DataCopy(queryKey, queryKeyGm_[source], width_);
            DataCopy(value, valueGm_[source], width_);
            pipe_barrier(PIPE_ALL);
            DataCopy(sortedQueryKeyGm_[output], queryKey, width_);
            DataCopy(sortedValueGm_[output], value, width_);
            pipe_barrier(PIPE_ALL);
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> queryKeyBuffer_;
    TBuf<TPosition::VECCALC> valueBuffer_;
    GlobalTensor<float> queryKeyGm_;
    GlobalTensor<float> valueGm_;
    GlobalTensor<int64_t> indicesGm_;
    GlobalTensor<float> sortedQueryKeyGm_;
    GlobalTensor<float> sortedValueGm_;
    uint32_t rows_ = 0U;
    uint32_t sourceLength_ = 0U;
    uint32_t indexLength_ = 0U;
    uint32_t width_ = 0U;
};

}  // namespace

extern "C" __global__ __aicore__ void reformer_lsh_qkv_gather(GM_ADDR queryKey, GM_ADDR value, GM_ADDR indices,
                                                              GM_ADDR sortedQueryKey, GM_ADDR sortedValue,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    const auto* config = reinterpret_cast<const LshGatherTiling*>(&tilingData);
    LshQkvGatherKernel kernel;
    kernel.Init(queryKey, value, indices, sortedQueryKey, sortedValue, *config);
    kernel.Process();
}
