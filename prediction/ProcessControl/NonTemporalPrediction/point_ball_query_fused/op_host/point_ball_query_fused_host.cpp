/** Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0. */
#include "point_ball_query_fused_host.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "../point_ball_query_tiling.h"
#include "acl/acl.h"
namespace
{
constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
constexpr uint32_t kCoreLimit = 40U;
uint64_t WorkspaceSize() { return (sizeof(BallQueryTiling) + 31U) / 32U * 32U; }
bool FitsUint32(int64_t value) { return value > 0 && value <= kUint32Max; }
}  // namespace
extern "C" uint32_t aclrtlaunch_point_ball_query_fused_kernel(uint32_t, aclrtStream, void*, void*, void*, void*, void*,
                                                              void*);
extern "C" uint64_t aclnnPointBallQueryFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t)
{
    return WorkspaceSize();
}
extern "C" int32_t aclnnPointBallQueryFused(void* points, void* queries, void* indices, void* counts, int64_t batch,
                                            int64_t point_count, int64_t query_count, int64_t sample_count,
                                            float radius, void* workspace, uint64_t workspace_size, void* stream)
{
    const uint64_t required = WorkspaceSize();
    if (points == nullptr || queries == nullptr || indices == nullptr || counts == nullptr || workspace == nullptr ||
        stream == nullptr || workspace_size < required || !FitsUint32(batch) || !FitsUint32(point_count) ||
        point_count > 8192 || !FitsUint32(query_count) || query_count > 1024 || !FitsUint32(sample_count) ||
        sample_count > 128 || batch > kUint32Max / query_count || !std::isfinite(radius) || radius <= 0.0f)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t total_queries = static_cast<uint32_t>(batch * query_count);
    BallQueryTiling tiling{static_cast<uint32_t>(batch),
                           static_cast<uint32_t>(point_count),
                           static_cast<uint32_t>(query_count),
                           static_cast<uint32_t>(sample_count),
                           total_queries,
                           std::min(total_queries, kCoreLimit),
                           radius * radius};
    const int32_t copy_ret = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copy_ret != ACL_SUCCESS) return copy_ret;
    const uint32_t ret = aclrtlaunch_point_ball_query_fused_kernel(
        tiling.coreNum, reinterpret_cast<aclrtStream>(stream), points, queries, indices, counts, workspace, workspace);
    return ret == 0U ? ACL_SUCCESS : static_cast<int32_t>(ret);
}
