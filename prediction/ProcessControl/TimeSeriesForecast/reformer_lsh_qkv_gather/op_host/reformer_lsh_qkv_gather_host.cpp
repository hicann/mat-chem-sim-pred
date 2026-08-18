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
#include "reformer_lsh_qkv_gather_tiling.h"

namespace
{
constexpr int64_t kMaxWidth = 16384;

bool FitsUint32(int64_t value)
{
    constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    return value > 0 && value <= kUint32Max;
}

bool CanMultiply(uint64_t lhs, uint64_t rhs) { return rhs == 0U || lhs <= std::numeric_limits<uint64_t>::max() / rhs; }
}  // namespace

namespace optiling
{
ge::graphStatus ReformerLshQkvGatherTiling(gert::TilingContext* context)
{
    const auto& source = context->GetInputShape(0)->GetStorageShape();
    const auto& second = context->GetInputShape(1)->GetStorageShape();
    const auto& indices = context->GetInputShape(2)->GetStorageShape();
    if (source.GetDimNum() != 3U || indices.GetDimNum() != 2U || !FitsUint32(source.GetDim(0)) ||
        !FitsUint32(source.GetDim(1)) || !FitsUint32(source.GetDim(2)) || indices.GetDim(0) != source.GetDim(0) ||
        !FitsUint32(indices.GetDim(1)) || source.GetDim(2) > kMaxWidth || source.GetDim(2) % 8 != 0)
    {
        return ge::GRAPH_FAILED;
    }
    if (second.GetDimNum() != 3U || second.GetDim(0) != source.GetDim(0) || second.GetDim(1) != source.GetDim(1) ||
        second.GetDim(2) != source.GetDim(2))
    {
        return ge::GRAPH_FAILED;
    }
    const uint64_t rows = static_cast<uint64_t>(source.GetDim(0));
    const uint64_t sourceLength = static_cast<uint64_t>(source.GetDim(1));
    const uint64_t indexLength = static_cast<uint64_t>(indices.GetDim(1));
    const uint64_t width = static_cast<uint64_t>(source.GetDim(2));
    if (!CanMultiply(rows, sourceLength))
    {
        return ge::GRAPH_FAILED;
    }
    const uint64_t sourceRows = rows * sourceLength;
    if (!CanMultiply(sourceRows, width))
    {
        return ge::GRAPH_FAILED;
    }
    if (!CanMultiply(rows, indexLength))
    {
        return ge::GRAPH_FAILED;
    }
    const uint64_t jobs = rows * indexLength;
    if (!CanMultiply(jobs, width))
    {
        return ge::GRAPH_FAILED;
    }
    ReformerLshQkvGatherTilingData tiling;
    tiling.set_rows(static_cast<uint32_t>(source.GetDim(0)));
    tiling.set_source_length(static_cast<uint32_t>(source.GetDim(1)));
    tiling.set_index_length(static_cast<uint32_t>(indices.GetDim(1)));
    tiling.set_width(static_cast<uint32_t>(source.GetDim(2)));
    const uint64_t maxUint32 = std::numeric_limits<uint32_t>::max();
    const uint32_t boundedJobs = static_cast<uint32_t>(jobs > maxUint32 ? maxUint32 : jobs);
    context->SetBlockDim(timeseries_host::SelectBlockDim(boundedJobs, timeseries_host::GetVectorCoreCount(context)));
    timeseries_host::StoreTilingData(context, tiling);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
graphStatus InferReformerLshQkvGatherShape(gert::InferShapeContext* context)
{
    const gert::Shape* source = context->GetInputShape(0);
    const gert::Shape* indices = context->GetInputShape(2);
    *context->GetOutputShape(0) = *source;
    context->GetOutputShape(0)->SetDim(1, indices->GetDim(1));
    *context->GetOutputShape(1) = *context->GetOutputShape(0);
    return ge::GRAPH_SUCCESS;
}

graphStatus InferReformerLshQkvGatherDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge
