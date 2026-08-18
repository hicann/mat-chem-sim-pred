/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_film_modulated_mean_fused_host.h"

#include <algorithm>
#include <cstdint>

#include "../../graph_message_host_common.h"
#include "acl/acl.h"

namespace
{
struct CsrFilmModulatedMeanFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t core_num;
    uint32_t max_segment_size;
    uint32_t apply_relu;
    uint32_t reserved[2];
};

bool ValidPointers(void* rowPtr, void* sourceIndex, void* projected, void* beta, void* gamma, void* output,
                   void* workspace, void* stream)
{
    return !GraphMessageHost::HasNull({rowPtr, sourceIndex, projected, beta, gamma, output, workspace, stream}) &&
           !GraphMessageHost::Aliases(output, {rowPtr, sourceIndex, projected, beta, gamma});
}

bool ValidDimensions(int64_t nodes, int64_t edges, int64_t channels, int64_t maxSegmentSize, int64_t applyRelu)
{
    return GraphMessageHost::PositiveInt32(nodes) && GraphMessageHost::PositiveInt32(edges) &&
           GraphMessageHost::InRange(channels, 512) && GraphMessageHost::InRange(maxSegmentSize, 2048) &&
           (applyRelu == 0 || applyRelu == 1);
}
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_film_modulated_mean_fused_kernel(uint32_t, aclrtStream, void*, void*, void*, void*,
                                                                     void*, void*, void*, void*);

extern "C" uint64_t aclnnCsrFilmModulatedMeanFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t)
{
    return GraphMessageHost::AlignUp32(sizeof(CsrFilmModulatedMeanFusedTiling));
}

extern "C" int32_t aclnnCsrFilmModulatedMeanFused(void* rowPtr, void* sourceIndex, void* projected, void* beta,
                                                  void* gamma, void* output, int64_t nodes, int64_t edges,
                                                  int64_t channels, int64_t maxSegmentSize, int64_t applyRelu,
                                                  void* workspace, uint64_t workspaceSize, void* stream)
{
    const uint64_t required = aclnnCsrFilmModulatedMeanFusedGetWorkspaceSize(nodes, edges, channels, maxSegmentSize);
    if (!ValidPointers(rowPtr, sourceIndex, projected, beta, gamma, output, workspace, stream) ||
        !ValidDimensions(nodes, edges, channels, maxSegmentSize, applyRelu) || workspaceSize < required)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    CsrFilmModulatedMeanFusedTiling tiling{static_cast<uint32_t>(nodes),
                                           static_cast<uint32_t>(edges),
                                           static_cast<uint32_t>(channels),
                                           cores,
                                           static_cast<uint32_t>(maxSegmentSize),
                                           static_cast<uint32_t>(applyRelu),
                                           {0U, 0U}};
    const int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    return static_cast<int32_t>(aclrtlaunch_csr_film_modulated_mean_fused_kernel(
        cores, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, projected, beta, gamma, output, workspace,
        workspace));
}
