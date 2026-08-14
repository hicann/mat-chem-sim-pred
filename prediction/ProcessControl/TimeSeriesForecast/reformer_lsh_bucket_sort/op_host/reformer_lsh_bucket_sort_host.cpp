/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>

#include "../../common/op_host_utils.h"
#include "reformer_lsh_bucket_sort_tiling.h"

namespace
{
bool FitsUint32(int64_t value)
{
    constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    return value > 0 && value <= kUint32Max;
}

bool FitsCountBuffer(int64_t value) { return value > 0 && value <= std::numeric_limits<int32_t>::max(); }
}  // namespace

namespace optiling
{
ge::graphStatus ReformerLshBucketSortTiling(gert::TilingContext* context)
{
    const auto& keys = context->GetInputShape(0)->GetStorageShape();
    if (keys.GetDimNum() != 2U || !FitsUint32(keys.GetDim(0)) || !FitsCountBuffer(keys.GetDim(1)))
    {
        return ge::GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const int64_t* sequenceLength = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* totalBuckets = attrs->GetAttrPointer<int64_t>(1);
    if (sequenceLength == nullptr || totalBuckets == nullptr || !FitsUint32(*sequenceLength) || *totalBuckets <= 0 ||
        *totalBuckets > 4096)
    {
        return ge::GRAPH_FAILED;
    }
    ReformerLshBucketSortTilingData tiling;
    tiling.set_rows(static_cast<uint32_t>(keys.GetDim(0)));
    tiling.set_total_length(static_cast<uint32_t>(keys.GetDim(1)));
    tiling.set_sequence_length(static_cast<uint32_t>(*sequenceLength));
    tiling.set_total_buckets(static_cast<uint32_t>(*totalBuckets));
    const uint32_t coreCount = timeseries_host::GetVectorCoreCount(context);
    context->SetBlockDim(timeseries_host::SelectBlockDim(static_cast<uint32_t>(keys.GetDim(0)), coreCount));
    timeseries_host::StoreTilingData(context, tiling);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
graphStatus InferReformerLshBucketSortShape(gert::InferShapeContext* context)
{
    const gert::Shape* input = context->GetInputShape(0);
    for (uint32_t i = 0; i < 3U; ++i)
    {
        *context->GetOutputShape(i) = *input;
    }
    return ge::GRAPH_SUCCESS;
}

graphStatus InferReformerLshBucketSortDataType(gert::InferDataTypeContext* context)
{
    for (uint32_t i = 0; i < 3U; ++i)
    {
        context->SetOutputDataType(i, context->GetInputDataType(0));
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge
