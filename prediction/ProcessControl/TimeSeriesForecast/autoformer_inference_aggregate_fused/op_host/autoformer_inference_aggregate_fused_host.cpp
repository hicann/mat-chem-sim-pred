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
#include "autoformer_inference_aggregate_fused_tiling.h"

namespace
{
constexpr int64_t kMaxSequenceLength = 4096;

bool FitsUint32(int64_t value)
{
    constexpr int64_t kUint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    return value > 0 && value <= kUint32Max;
}

bool CanMultiply(uint64_t lhs, uint64_t rhs) { return rhs == 0U || lhs <= std::numeric_limits<uint64_t>::max() / rhs; }
}  // namespace

namespace optiling
{
ge::graphStatus AutoformerInferenceAggregateFusedTiling(gert::TilingContext* context)
{
    const auto& values = context->GetInputShape(0)->GetStorageShape();
    const auto& correlation = context->GetInputShape(1)->GetStorageShape();
    if (values.GetDimNum() != 4U || correlation.GetDimNum() != 4U || !FitsUint32(values.GetDim(0)) ||
        !FitsUint32(values.GetDim(1)) || !FitsUint32(values.GetDim(2)) || !FitsUint32(values.GetDim(3)) ||
        values.GetDim(3) > kMaxSequenceLength || values.GetDim(3) % 8 != 0)
    {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < 4U; ++i)
    {
        if (correlation.GetDim(i) != values.GetDim(i))
        {
            return ge::GRAPH_FAILED;
        }
    }
    uint64_t elements = 1U;
    for (uint32_t i = 0; i < 4U; ++i)
    {
        const uint64_t dim = static_cast<uint64_t>(values.GetDim(i));
        if (!CanMultiply(elements, dim))
        {
            return ge::GRAPH_FAILED;
        }
        elements *= dim;
    }
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const int64_t* topK = attrs->GetAttrPointer<int64_t>(0);
    if (topK == nullptr || *topK <= 0 || *topK > 16 || *topK > values.GetDim(3))
    {
        return ge::GRAPH_FAILED;
    }
    AutoformerInferenceAggregateFusedTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(values.GetDim(0)));
    tiling.set_heads(static_cast<uint32_t>(values.GetDim(1)));
    tiling.set_channels(static_cast<uint32_t>(values.GetDim(2)));
    tiling.set_length(static_cast<uint32_t>(values.GetDim(3)));
    tiling.set_top_k(static_cast<uint32_t>(*topK));
    tiling.set_inverse_heads(1.0F / static_cast<float>(values.GetDim(1)));
    tiling.set_inverse_channels(1.0F / static_cast<float>(values.GetDim(2)));
    const uint32_t coreCount = timeseries_host::GetVectorCoreCount(context);
    context->SetBlockDim(timeseries_host::SelectBlockDim(static_cast<uint32_t>(values.GetDim(0)), coreCount));
    timeseries_host::StoreTilingData(context, tiling);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
graphStatus InferAutoformerInferenceAggregateFusedShape(gert::InferShapeContext* context)
{
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return ge::GRAPH_SUCCESS;
}

graphStatus InferAutoformerInferenceAggregateFusedDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge
