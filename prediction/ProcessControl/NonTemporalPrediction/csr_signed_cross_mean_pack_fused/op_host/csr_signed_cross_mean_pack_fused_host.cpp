/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_signed_cross_mean_pack_fused_host.h"

#include <algorithm>
#include <cstdint>

#include "../../graph_message_host_common.h"
#include "acl/acl.h"

namespace
{
struct Tiling
{
    uint32_t nodes, positive_edges, negative_edges, channels;
    uint32_t core_num, max_segment_size, reserved0, reserved1;
};
bool ValidPointers(void* positiveRowPtr, void* positiveSourceIndex, void* negativeRowPtr, void* negativeSourceIndex,
                   void* features, void* positiveInverseDegree, void* negativeInverseDegree, void* output,
                   void* workspace, void* stream)
{
    return !GraphMessageHost::HasNull({positiveRowPtr, positiveSourceIndex, negativeRowPtr, negativeSourceIndex,
                                       features, positiveInverseDegree, negativeInverseDegree, output, workspace,
                                       stream}) &&
           !GraphMessageHost::Aliases(output, {features, positiveInverseDegree, negativeInverseDegree});
}

bool ValidDimensions(int64_t nodes, int64_t positiveEdges, int64_t negativeEdges, int64_t channels,
                     int64_t maxSegmentSize)
{
    return GraphMessageHost::PositiveInt32(nodes) && GraphMessageHost::PositiveInt32(positiveEdges) &&
           GraphMessageHost::PositiveInt32(negativeEdges) && GraphMessageHost::InRange(channels, 64) &&
           channels % 8 == 0 && GraphMessageHost::InRange(maxSegmentSize, 1024);
}
}  // namespace
extern "C" uint32_t aclrtlaunch_csr_signed_cross_mean_pack_fused_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                        void*, void*, void*, void*, void*, void*,
                                                                        void*);

extern "C" uint64_t aclnnCsrSignedCrossMeanPackFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t, int64_t)
{
    return GraphMessageHost::AlignUp32(sizeof(Tiling));
}

extern "C" int32_t aclnnCsrSignedCrossMeanPackFused(void* positiveRowPtr, void* positiveSourceIndex,
                                                    void* negativeRowPtr, void* negativeSourceIndex, void* features,
                                                    void* positiveInverseDegree, void* negativeInverseDegree,
                                                    void* output, int64_t nodes, int64_t positiveEdges,
                                                    int64_t negativeEdges, int64_t channels, int64_t maxSegmentSize,
                                                    void* workspace, uint64_t workspaceSize, void* stream)
{
    uint64_t required =
        aclnnCsrSignedCrossMeanPackFusedGetWorkspaceSize(nodes, positiveEdges, negativeEdges, channels, maxSegmentSize);
    if (!ValidPointers(positiveRowPtr, positiveSourceIndex, negativeRowPtr, negativeSourceIndex, features,
                       positiveInverseDegree, negativeInverseDegree, output, workspace, stream) ||
        !ValidDimensions(nodes, positiveEdges, negativeEdges, channels, maxSegmentSize) || workspaceSize < required)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    Tiling tiling{static_cast<uint32_t>(nodes),
                  static_cast<uint32_t>(positiveEdges),
                  static_cast<uint32_t>(negativeEdges),
                  static_cast<uint32_t>(channels),
                  cores,
                  static_cast<uint32_t>(maxSegmentSize),
                  0U,
                  0U};
    int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS) return copied;
    return static_cast<int32_t>(aclrtlaunch_csr_signed_cross_mean_pack_fused_kernel(
        cores, reinterpret_cast<aclrtStream>(stream), positiveRowPtr, positiveSourceIndex, negativeRowPtr,
        negativeSourceIndex, features, positiveInverseDegree, negativeInverseDegree, output, workspace, workspace));
}
