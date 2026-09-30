/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_genconv_softmax_aggregate_fused_host.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "acl/acl.h"

namespace
{
struct CsrGenConvSoftmaxAggregateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t core_num;
    float temperature;
};

uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_genconv_softmax_aggregate_fused_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                           void*, void*, void*);

extern "C" uint64_t aclnnCsrGenConvSoftmaxAggregateFusedGetWorkspaceSize(int64_t, int64_t, int64_t, float)
{
    return AlignUp(sizeof(CsrGenConvSoftmaxAggregateFusedTiling));
}

extern "C" int32_t aclnnCsrGenConvSoftmaxAggregateFused(void* rowPtr, void* sourceIndex, void* features, void* output,
                                                        int64_t nodes, int64_t edges, int64_t channels,
                                                        float temperature, void* workspace, uint64_t workspaceSize,
                                                        void* stream)
{
    const uint64_t required = aclnnCsrGenConvSoftmaxAggregateFusedGetWorkspaceSize(nodes, edges, channels, temperature);
    if (rowPtr == nullptr || sourceIndex == nullptr || features == nullptr || output == nullptr ||
        workspace == nullptr || stream == nullptr || workspaceSize < required || nodes <= 0 || nodes > INT32_MAX ||
        edges <= 0 || edges > INT32_MAX || channels <= 0 || channels > 1024 || !std::isfinite(temperature))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t coreNum = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    CsrGenConvSoftmaxAggregateFusedTiling tiling{static_cast<uint32_t>(nodes), static_cast<uint32_t>(edges),
                                                 static_cast<uint32_t>(channels), coreNum, temperature};
    const int32_t copyResult =
        aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyResult != ACL_SUCCESS)
    {
        return copyResult;
    }
    return static_cast<int32_t>(aclrtlaunch_csr_genconv_softmax_aggregate_fused_kernel(
        coreNum, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, features, output, workspace, workspace));
}
