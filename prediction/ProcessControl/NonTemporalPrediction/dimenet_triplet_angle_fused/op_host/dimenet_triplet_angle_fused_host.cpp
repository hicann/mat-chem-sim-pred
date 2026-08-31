/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "dimenet_triplet_angle_fused_host.h"

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"

extern "C" uint32_t aclrtlaunch_dimenet_triplet_angle_fused_kernel(uint32_t, aclrtStream, void*, void*, void*, void*,
                                                                   void*, uint32_t, uint32_t);

extern "C" uint64_t aclnnDimeNetTripletAngleFusedGetWorkspaceSize(int64_t nodes, int64_t triplets)
{
    (void)nodes;
    (void)triplets;
    return 0U;
}

extern "C" int32_t aclnnDimeNetTripletAngleFused(void* position, void* idxI, void* idxJ, void* idxK, void* angle,
                                                 int64_t nodes, int64_t triplets, void* workspace,
                                                 uint64_t workspaceSize, void* stream)
{
    if (position == nullptr || idxI == nullptr || idxJ == nullptr || idxK == nullptr || angle == nullptr ||
        stream == nullptr || workspaceSize != 0U || nodes <= 0 || nodes > INT32_MAX || triplets <= 0 ||
        triplets > INT32_MAX)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    (void)workspace;
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(triplets, 40)));
    return static_cast<int32_t>(aclrtlaunch_dimenet_triplet_angle_fused_kernel(
        cores, reinterpret_cast<aclrtStream>(stream), position, idxI, idxJ, idxK, angle, static_cast<uint32_t>(nodes),
        static_cast<uint32_t>(triplets)));
}
