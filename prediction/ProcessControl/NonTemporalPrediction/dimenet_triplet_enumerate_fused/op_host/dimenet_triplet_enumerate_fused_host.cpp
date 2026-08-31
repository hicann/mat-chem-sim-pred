/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "dimenet_triplet_enumerate_fused_host.h"

#include <cstdint>

#include "acl/acl.h"

namespace
{
struct DimeNetTripletEnumerateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t capacity;
    uint32_t reserved;
};

uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }

bool ValidPointers(void* const* pointers, uint32_t count, void* workspace)
{
    for (uint32_t left = 0U; left < count; ++left)
    {
        if (workspace == pointers[left])
        {
            return false;
        }
        for (uint32_t right = left + 1U; right < count; ++right)
        {
            if (pointers[left] == pointers[right])
            {
                return false;
            }
        }
    }
    return true;
}

bool ValidParameters(int64_t nodes, int64_t edges, int64_t capacity, void* workspace, uint64_t workspaceSize,
                     void* const* pointers)
{
    const uint64_t required = aclnnDimeNetTripletEnumerateFusedGetWorkspaceSize(nodes, edges, capacity);
    const bool sizes = nodes > 0 && nodes <= INT32_MAX && edges > 0 && edges <= INT32_MAX && capacity > 0 &&
                       capacity <= INT32_MAX && workspaceSize >= required;
    return workspace != nullptr && sizes && ValidPointers(pointers, 8U, workspace);
}
}  // namespace

extern "C" uint32_t aclrtlaunch_dimenet_triplet_enumerate_fused_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                       void*, void*, void*, void*, void*, void*, void*);

extern "C" uint64_t aclnnDimeNetTripletEnumerateFusedGetWorkspaceSize(int64_t nodes, int64_t edges, int64_t capacity)
{
    (void)nodes;
    (void)edges;
    (void)capacity;
    return AlignUp(sizeof(DimeNetTripletEnumerateFusedTiling));
}

extern "C" int32_t aclnnDimeNetTripletEnumerateFused(void* rowPtr, void* sourceIndex, void* idxI, void* idxJ,
                                                     void* idxK, void* idxKj, void* idxJi, void* counts, int64_t nodes,
                                                     int64_t edges, int64_t capacity, void* workspace,
                                                     uint64_t workspaceSize, void* stream)
{
    void* pointers[] = {rowPtr, sourceIndex, idxI, idxJ, idxK, idxKj, idxJi, counts};
    if (stream == nullptr || !ValidParameters(nodes, edges, capacity, workspace, workspaceSize, pointers))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const DimeNetTripletEnumerateFusedTiling tiling{static_cast<uint32_t>(nodes), static_cast<uint32_t>(edges),
                                                    static_cast<uint32_t>(capacity), 0U};
    const int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    return static_cast<int32_t>(aclrtlaunch_dimenet_triplet_enumerate_fused_kernel(
        1U, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, idxI, idxJ, idxK, idxKj, idxJi, counts,
        workspace, workspace));
}
