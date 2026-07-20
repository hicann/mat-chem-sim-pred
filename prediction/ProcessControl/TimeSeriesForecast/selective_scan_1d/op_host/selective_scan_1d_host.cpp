/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "selective_scan_1d_host.h"

#include <algorithm>
#include <limits>

#include "acl/acl.h"

namespace {

struct SelectiveScanTilingData {
    uint32_t batch;
    uint32_t length;
    uint32_t dim;
    uint32_t state;
};

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return ((value + align - 1U) / align) * align;
}

bool FitsUint32(int64_t value)
{
    constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    return value >= 0 && value <= kUint32Max;
}

bool HasValidShape(int64_t batch, int64_t length, int64_t dim, int64_t state)
{
    return batch > 0 && length > 0 && dim > 0 && state > 0 && FitsUint32(batch) && FitsUint32(length) &&
           FitsUint32(dim) && FitsUint32(state);
}

uint32_t ComputeBlockDim(int64_t batch, int64_t dim)
{
    const int64_t groups = batch * dim;
    return static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(32, groups)));
}

}  // namespace

extern "C" void aclrtlaunch_selective_scan1_d(
    uint32_t blockDim, aclrtStream stream, void* u, void* delta, void* a, void* b, void* c, void* d, void* output,
    void* kernel_workspace, void* tiling);

extern "C" int32_t aclnnSelectiveScan1D(
    void* u, void* delta, void* a, void* b, void* c, void* d, void* output, int64_t batch, int64_t length,
    int64_t dim, int64_t state, void* workspace, uint64_t workspace_size, void* stream)
{
    if (u == nullptr || delta == nullptr || a == nullptr || b == nullptr || c == nullptr || d == nullptr ||
        output == nullptr || workspace == nullptr || stream == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (!HasValidShape(batch, length, dim, state) ||
        workspace_size < aclnnSelectiveScan1DGetWorkspaceSize(batch, length, dim, state)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    SelectiveScanTilingData tiling;
    tiling.batch = static_cast<uint32_t>(batch);
    tiling.length = static_cast<uint32_t>(length);
    tiling.dim = static_cast<uint32_t>(dim);
    tiling.state = static_cast<uint32_t>(state);

    const auto ret = aclrtMemcpyAsync(
        workspace, sizeof(SelectiveScanTilingData), &tiling, sizeof(SelectiveScanTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE, reinterpret_cast<aclrtStream>(stream));
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    aclrtlaunch_selective_scan1_d(
        ComputeBlockDim(batch, dim), reinterpret_cast<aclrtStream>(stream), u, delta, a, b, c, d, output, nullptr,
        workspace);
    return ACL_SUCCESS;
}

extern "C" uint64_t aclnnSelectiveScan1DGetWorkspaceSize(
    int64_t batch, int64_t length, int64_t dim, int64_t state)
{
    (void)batch;
    (void)length;
    (void)dim;
    (void)state;
    return AlignUp(sizeof(SelectiveScanTilingData), 32U);
}
