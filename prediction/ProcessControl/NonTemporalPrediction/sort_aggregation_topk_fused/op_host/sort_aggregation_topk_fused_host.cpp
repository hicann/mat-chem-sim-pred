/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "sort_aggregation_topk_fused_host.h"

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"

namespace
{
struct SortTiling
{
    // Sort host layout is consumed by the top-k kernel.
    static constexpr uint32_t kLayoutTag = 0x534F5254U;
    uint32_t nodes;
    uint32_t graphs;
    uint32_t channels;
    uint32_t topK;
    uint32_t coreNum;
    uint32_t reserved[3];
};

uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }
}  // namespace

extern "C" uint32_t aclrtlaunch_sort_aggregation_topk_fused_kernel(uint32_t, aclrtStream, void*, void*, void*, void*,
                                                                   void*);

extern "C" uint64_t aclnnSortAggregationTopkFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t)
{
    return AlignUp(sizeof(SortTiling));
}

extern "C" int32_t aclnnSortAggregationTopkFused(void* graphPtr, void* features, void* output, int64_t nodes,
                                                 int64_t graphs, int64_t channels, int64_t topK, void* workspace,
                                                 uint64_t workspaceSize, void* stream)
{
    const uint64_t required = aclnnSortAggregationTopkFusedGetWorkspaceSize(nodes, graphs, channels, topK);
    if (graphPtr == nullptr || features == nullptr || output == nullptr || workspace == nullptr || stream == nullptr ||
        workspaceSize < required || nodes <= 0 || nodes > UINT32_MAX || graphs <= 0 || graphs > UINT32_MAX ||
        channels <= 0 || channels > 1024 || topK <= 0 || topK > 128)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t coreCount = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(graphs, 40)));
    SortTiling tiling{static_cast<uint32_t>(nodes),
                      static_cast<uint32_t>(graphs),
                      static_cast<uint32_t>(channels),
                      static_cast<uint32_t>(topK),
                      coreCount,
                      {0U, 0U, 0U}};
    const int32_t copyRet = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyRet != ACL_SUCCESS) return copyRet;
    return static_cast<int32_t>(aclrtlaunch_sort_aggregation_topk_fused_kernel(
        tiling.coreNum, reinterpret_cast<aclrtStream>(stream), graphPtr, features, output, workspace, workspace));
}
