/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "csr_sage_mean_root_relu_fused_host.h"

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"

namespace
{
struct CsrSageMeanRootReluTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t coreNum;
};

uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_sage_mean_root_relu_fused_kernel(uint32_t, aclrtStream, void*, void*, void*, void*,
                                                                     void*, void*, void*);

extern "C" uint64_t aclnnCsrSageMeanRootReluFusedGetWorkspaceSize(int64_t, int64_t, int64_t)
{
    return AlignUp(sizeof(CsrSageMeanRootReluTiling));
}

extern "C" int32_t aclnnCsrSageMeanRootReluFused(void* rowPtr, void* columnIndex, void* neighborFeatures,
                                                 void* rootFeatures, void* output, int64_t nodes, int64_t edges,
                                                 int64_t channels, void* workspace, uint64_t workspaceSize,
                                                 void* stream)
{
    const uint64_t required = aclnnCsrSageMeanRootReluFusedGetWorkspaceSize(nodes, edges, channels);
    if (rowPtr == nullptr || columnIndex == nullptr || neighborFeatures == nullptr || rootFeatures == nullptr ||
        output == nullptr || workspace == nullptr || stream == nullptr || workspaceSize < required || nodes <= 0 ||
        nodes > UINT32_MAX || edges <= 0 || edges > UINT32_MAX || channels <= 0 || channels > 1024 || channels % 2 != 0)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t coreCount = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    CsrSageMeanRootReluTiling tiling{static_cast<uint32_t>(nodes), static_cast<uint32_t>(edges),
                                     static_cast<uint32_t>(channels), coreCount};
    const int32_t copyResult =
        aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyResult != ACL_SUCCESS)
    {
        return copyResult;
    }
    return static_cast<int32_t>(aclrtlaunch_csr_sage_mean_root_relu_fused_kernel(
        tiling.coreNum, reinterpret_cast<aclrtStream>(stream), rowPtr, columnIndex, neighborFeatures, rootFeatures,
        output, workspace, workspace));
}
