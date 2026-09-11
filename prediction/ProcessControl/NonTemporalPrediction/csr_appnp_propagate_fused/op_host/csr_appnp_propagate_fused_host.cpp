/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "csr_appnp_propagate_fused_host.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "csr_appnp_propagate_fused_tiling.h"

namespace
{
uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }

bool TensorBytes(int64_t nodes, int64_t channels, uint64_t& bytes)
{
    if (nodes <= 0 || channels <= 0 ||
        static_cast<uint64_t>(nodes) >
            std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(channels) / sizeof(float))
    {
        return false;
    }
    bytes = static_cast<uint64_t>(nodes) * static_cast<uint64_t>(channels) * sizeof(float);
    return true;
}

bool ValidArguments(void* rowPtr, void* columnIndex, void* edgeWeight, void* initial, void* output, int64_t nodes,
                    int64_t edges, int64_t channels, int64_t iterations, float alpha, void* workspace,
                    uint64_t workspaceSize, void* stream, uint64_t required)
{
    return rowPtr != nullptr && columnIndex != nullptr && edgeWeight != nullptr && initial != nullptr &&
           output != nullptr && output != initial && workspace != nullptr && stream != nullptr && nodes > 0 &&
           nodes <= UINT32_MAX && edges > 0 && edges <= UINT32_MAX && channels > 0 && channels <= 1024 &&
           iterations > 0 && iterations <= 64 && std::isfinite(alpha) && alpha >= 0.0F && alpha <= 1.0F &&
           workspaceSize >= required;
}

int32_t LaunchIterations(void* rowPtr, void* columnIndex, void* edgeWeight, void* initial, void* output,
                         int64_t iterations, uint64_t tensorBytes, const CsrAppnpPropagateFusedTiling& tiling,
                         void* workspace, void* stream)
{
    auto* scratch = reinterpret_cast<uint8_t*>(workspace) + AlignUp(sizeof(CsrAppnpPropagateFusedTiling));
    void* current = initial;
    for (int64_t iteration = 0; iteration < iterations; ++iteration)
    {
        void* destination = iteration + 1 == iterations
                                ? output
                                : static_cast<void*>(scratch + static_cast<uint64_t>(iteration & 1) * tensorBytes);
        const uint32_t launchRet = aclrtlaunch_csr_appnp_propagate_fused_kernel(
            tiling.coreNum, reinterpret_cast<aclrtStream>(stream), rowPtr, columnIndex, edgeWeight, current, initial,
            destination, workspace, workspace);
        if (launchRet != ACL_SUCCESS) return static_cast<int32_t>(launchRet);
        current = destination;
    }
    return ACL_SUCCESS;
}
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_appnp_propagate_fused_kernel(uint32_t, aclrtStream, void*, void*, void*, void*,
                                                                 void*, void*, void*, void*);

extern "C" uint64_t aclnnCsrAppnpPropagateFusedGetWorkspaceSize(int64_t nodes, int64_t, int64_t channels, int64_t)
{
    uint64_t tensorBytes = 0U;
    if (!TensorBytes(nodes, channels, tensorBytes) ||
        tensorBytes > (std::numeric_limits<uint64_t>::max() - AlignUp(sizeof(Tiling))) / 2U)
    {
        return AlignUp(sizeof(Tiling));
    }
    return AlignUp(sizeof(Tiling)) + 2U * tensorBytes;
}

extern "C" int32_t aclnnCsrAppnpPropagateFused(void* rowPtr, void* columnIndex, void* edgeWeight, void* initial,
                                               void* output, int64_t nodes, int64_t edges, int64_t channels,
                                               int64_t iterations, float alpha, void* workspace, uint64_t workspaceSize,
                                               void* stream)
{
    uint64_t tensorBytes = 0U;
    const uint64_t required = aclnnCsrAppnpPropagateFusedGetWorkspaceSize(nodes, edges, channels, iterations);
    if (!TensorBytes(nodes, channels, tensorBytes) ||
        !ValidArguments(rowPtr, columnIndex, edgeWeight, initial, output, nodes, edges, channels, iterations, alpha,
                        workspace, workspaceSize, stream, required))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t coreCount = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    CsrAppnpPropagateFusedTiling tiling{static_cast<uint32_t>(nodes),
                                        static_cast<uint32_t>(edges),
                                        static_cast<uint32_t>(channels),
                                        coreCount,
                                        alpha,
                                        {0U, 0U, 0U}};
    const int32_t copyRet = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyRet != ACL_SUCCESS) return copyRet;
    return LaunchIterations(rowPtr, columnIndex, edgeWeight, initial, output, iterations, tensorBytes, tiling,
                            workspace, stream);
}
