/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
struct DimeNetTripletEnumerateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t capacity;
    uint32_t reserved;
};

class DimeNetTripletEnumerateFusedKernel
{
   public:
    __aicore__ inline DimeNetTripletEnumerateFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR sourceIndex, GM_ADDR idxI, GM_ADDR idxJ, GM_ADDR idxK,
                                GM_ADDR idxKj, GM_ADDR idxJi, GM_ADDR counts,
                                const __gm__ DimeNetTripletEnumerateFusedTiling* tiling)
    {
        nodes_ = tiling->nodes;
        edges_ = tiling->edges;
        capacity_ = tiling->capacity;
        rowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr), nodes_ + 1U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        idxIGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxI), capacity_);
        idxJGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxJ), capacity_);
        idxKGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxK), capacity_);
        idxKjGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxKj), capacity_);
        idxJiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxJi), capacity_);
        countsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts), 2U);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0U)
        {
            return;
        }
        uint32_t written = 0U;
        uint32_t overflow = 0U;
        for (uint32_t target = 0U; target < nodes_; ++target)
        {
            EnumerateTarget(target, written, overflow);
        }
        countsGm_.SetValue(0U, static_cast<int32_t>(written));
        countsGm_.SetValue(1U, static_cast<int32_t>(overflow));
    }

   private:
    __aicore__ inline void EnumerateTarget(uint32_t target, uint32_t& written, uint32_t& overflow)
    {
        const int32_t edgeBegin = rowPtrGm_.GetValue(target);
        const int32_t edgeEnd = rowPtrGm_.GetValue(target + 1U);
        if (edgeBegin < 0 || edgeEnd < edgeBegin || static_cast<uint32_t>(edgeEnd) > edges_)
        {
            overflow = 1U;
            return;
        }
        for (int32_t edgeJi = edgeBegin; edgeJi < edgeEnd; ++edgeJi)
        {
            EnumerateMiddle(target, edgeJi, written, overflow);
        }
    }

    __aicore__ inline void EnumerateMiddle(uint32_t target, int32_t edgeJi, uint32_t& written, uint32_t& overflow)
    {
        const int32_t middle = sourceIndexGm_.GetValue(edgeJi);
        if (middle < 0 || static_cast<uint32_t>(middle) >= nodes_)
        {
            overflow = 1U;
            return;
        }
        const int32_t begin = rowPtrGm_.GetValue(middle);
        const int32_t end = rowPtrGm_.GetValue(middle + 1);
        if (begin < 0 || end < begin || static_cast<uint32_t>(end) > edges_)
        {
            overflow = 1U;
            return;
        }
        for (int32_t edgeKj = begin; edgeKj < end; ++edgeKj)
        {
            WriteTriplet(target, middle, edgeJi, edgeKj, written, overflow);
        }
    }

    __aicore__ inline void WriteTriplet(uint32_t target, int32_t middle, int32_t edgeJi, int32_t edgeKj,
                                        uint32_t& written, uint32_t& overflow)
    {
        const int32_t source = sourceIndexGm_.GetValue(edgeKj);
        if (source == static_cast<int32_t>(target))
        {
            return;
        }
        if (source < 0 || static_cast<uint32_t>(source) >= nodes_ || written >= capacity_)
        {
            overflow = 1U;
            return;
        }
        idxIGm_.SetValue(written, static_cast<int32_t>(target));
        idxJGm_.SetValue(written, middle);
        idxKGm_.SetValue(written, source);
        idxKjGm_.SetValue(written, edgeKj);
        idxJiGm_.SetValue(written, edgeJi);
        ++written;
    }

    GlobalTensor<int32_t> rowPtrGm_, sourceIndexGm_;
    GlobalTensor<int32_t> idxIGm_, idxJGm_, idxKGm_, idxKjGm_, idxJiGm_;
    GlobalTensor<int32_t> countsGm_;
    uint32_t nodes_ = 0U;
    uint32_t edges_ = 0U;
    uint32_t capacity_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void dimenet_triplet_enumerate_fused_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                             GM_ADDR idxI, GM_ADDR idxJ, GM_ADDR idxK,
                                                                             GM_ADDR idxKj, GM_ADDR idxJi,
                                                                             GM_ADDR counts, GM_ADDR workspace,
                                                                             GM_ADDR tiling)
{
    (void)workspace;
    DimeNetTripletEnumerateFusedKernel op;
    op.Init(rowPtr, sourceIndex, idxI, idxJ, idxK, idxKj, idxJi, counts,
            reinterpret_cast<const __gm__ DimeNetTripletEnumerateFusedTiling*>(tiling));
    op.Process();
}
