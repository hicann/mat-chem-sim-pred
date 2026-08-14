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

struct LshBucketSortTiling
{
    uint32_t rows;
    uint32_t total_length;
    uint32_t sequence_length;
    uint32_t total_buckets;
};

class LshBucketSortKernel
{
   public:
    __aicore__ inline LshBucketSortKernel() = default;

    __aicore__ inline void Init(GM_ADDR keys, GM_ADDR sortedKeys, GM_ADDR sticker, GM_ADDR inverse, GM_ADDR tiling)
    {
        const __gm__ LshBucketSortTiling* tilingData = reinterpret_cast<const __gm__ LshBucketSortTiling*>(tiling);
        rows_ = tilingData->rows;
        totalLength_ = tilingData->total_length;
        sequenceLength_ = tilingData->sequence_length;
        totalBuckets_ = tilingData->total_buckets;
        const uint64_t elements = static_cast<uint64_t>(rows_) * totalLength_;
        keysGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(keys), elements);
        sortedKeysGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(sortedKeys), elements);
        stickerGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(sticker), elements);
        inverseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(inverse), elements);
        pipe_.InitBuffer(countBuffer_, ((totalBuckets_ + 7U) / 8U) * 8U * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = GetBlockIdx();
        const uint32_t blocks = GetBlockNum();
        LocalTensor<int32_t> counts = countBuffer_.Get<int32_t>();
        for (uint64_t row = block; row < rows_; row += blocks)
        {
            for (uint32_t bucket = 0; bucket < totalBuckets_; ++bucket)
            {
                counts.SetValue(bucket, 0);
            }
            pipe_barrier(PIPE_ALL);
            const uint64_t base = row * totalLength_;
            for (uint32_t position = 0; position < totalLength_; ++position)
            {
                const int64_t key = keysGm_.GetValue(base + position);
                const uint32_t bucket = static_cast<uint32_t>(key / sequenceLength_);
                counts.SetValue(bucket, counts.GetValue(bucket) + 1);
            }
            pipe_barrier(PIPE_ALL);
            int32_t running = 0;
            for (uint32_t bucket = 0; bucket < totalBuckets_; ++bucket)
            {
                const int32_t count = counts.GetValue(bucket);
                counts.SetValue(bucket, running);
                running += count;
            }
            pipe_barrier(PIPE_ALL);
            for (uint32_t position = 0; position < totalLength_; ++position)
            {
                const int64_t key = keysGm_.GetValue(base + position);
                const uint32_t bucket = static_cast<uint32_t>(key / sequenceLength_);
                const int32_t sortedPosition = counts.GetValue(bucket);
                counts.SetValue(bucket, sortedPosition + 1);
                const uint64_t output = base + static_cast<uint32_t>(sortedPosition);
                sortedKeysGm_.SetValue(output, key);
                stickerGm_.SetValue(output, static_cast<int64_t>(position));
                inverseGm_.SetValue(base + position, static_cast<int64_t>(sortedPosition));
            }
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> countBuffer_;
    GlobalTensor<int64_t> keysGm_;
    GlobalTensor<int64_t> sortedKeysGm_;
    GlobalTensor<int64_t> stickerGm_;
    GlobalTensor<int64_t> inverseGm_;
    uint32_t rows_ = 0U;
    uint32_t totalLength_ = 0U;
    uint32_t sequenceLength_ = 0U;
    uint32_t totalBuckets_ = 0U;
};

}  // namespace

extern "C" __global__ __aicore__ void reformer_lsh_bucket_sort(GM_ADDR keys, GM_ADDR sortedKeys, GM_ADDR sticker,
                                                               GM_ADDR inverse, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    LshBucketSortKernel kernel;
    kernel.Init(keys, sortedKeys, sticker, inverse, tiling);
    kernel.Process();
}
