/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include "farthest_point_sampling_fused_host.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "acl/acl.h"

namespace
{

constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
constexpr uint32_t kCoreLimit = 40U;

struct FarthestPointTiling
{
    uint32_t batch;
    uint32_t pointCount;
    uint32_t sampleCount;
    uint32_t coreNum;
};

uint64_t WorkspaceSize() { return (sizeof(FarthestPointTiling) + 31U) / 32U * 32U; }

bool FitsUint32(int64_t value) { return value > 0 && value <= kUint32Max; }

bool Valid(void* workspace, void* stream, uint64_t workspace_size, uint64_t required)
{
    return workspace != nullptr && stream != nullptr && workspace_size >= required;
}

int32_t StoreTiling(void* workspace, const FarthestPointTiling& tiling)
{
    return aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
}

}  // namespace

extern "C" uint32_t aclrtlaunch_farthest_point_sampling_fused_kernel(uint32_t, aclrtStream, void*, void*, void*, void*);

extern "C" uint64_t aclnnFarthestPointSamplingFusedGetWorkspaceSize(int64_t, int64_t, int64_t)
{
    return WorkspaceSize();
}

extern "C" int32_t aclnnFarthestPointSamplingFused(void* points, void* sample_indices, int64_t batch,
                                                   int64_t point_count, int64_t sample_count, void* workspace,
                                                   uint64_t workspace_size, void* stream)
{
    const uint64_t required = aclnnFarthestPointSamplingFusedGetWorkspaceSize(batch, point_count, sample_count);
    if (points == nullptr || sample_indices == nullptr || !Valid(workspace, stream, workspace_size, required) ||
        !FitsUint32(batch) || point_count <= 1 || point_count > 4096 || sample_count <= 0 ||
        sample_count > point_count || sample_count > 512)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    FarthestPointTiling tiling{static_cast<uint32_t>(batch), static_cast<uint32_t>(point_count),
                               static_cast<uint32_t>(sample_count),
                               static_cast<uint32_t>(std::min<int64_t>(batch, kCoreLimit))};
    const int32_t copy_ret = StoreTiling(workspace, tiling);
    if (copy_ret != ACL_SUCCESS)
    {
        return copy_ret;
    }
    const uint32_t ret = aclrtlaunch_farthest_point_sampling_fused_kernel(
        tiling.coreNum, reinterpret_cast<aclrtStream>(stream), points, sample_indices, workspace, workspace);
    return ret == 0U ? ACL_SUCCESS : static_cast<int32_t>(ret);
}
