/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TIME_SERIES_FORECAST_OP_HOST_UTILS_H
#define TIME_SERIES_FORECAST_OP_HOST_UTILS_H

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace timeseries_host {

inline uint32_t SelectBlockDim(uint32_t workItems, uint32_t coreLimit)
{
    if (workItems == 0U || coreLimit == 0U) {
        return 1U;
    }
    return workItems < coreLimit ? workItems : coreLimit;
}

inline uint32_t GetVectorCoreCount(gert::TilingContext* context)
{
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    return platform.GetCoreNumAiv();
}

template <typename TilingData>
inline void StoreTilingData(gert::TilingContext* context, TilingData& tiling)
{
    auto* rawTiling = context->GetRawTilingData();
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());
}

inline ge::graphStatus CopyFirstInputShape(gert::InferShapeContext* context)
{
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CopyFirstInputDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferScanOutputShape(gert::InferShapeContext* context, uint32_t hiddenDivisor)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    outputShape->SetDim(inputShape->GetDimNum() - 1, weightShape->GetDim(1) / hiddenDivisor);
    return ge::GRAPH_SUCCESS;
}

template <typename TilingData>
inline ge::graphStatus BuildScanTiling(gert::TilingContext* context, uint32_t hiddenDivisor)
{
    const auto& inputShape = context->GetInputShape(0)->GetStorageShape();
    const auto& weightShape = context->GetInputShape(1)->GetStorageShape();
    const uint32_t batch = static_cast<uint32_t>(inputShape.GetDim(0));

    TilingData tiling;
    tiling.set_batch(batch);
    tiling.set_length(static_cast<uint32_t>(inputShape.GetDim(1)));
    tiling.set_in_size(static_cast<uint32_t>(inputShape.GetDim(2)));
    tiling.set_hidden(static_cast<uint32_t>(weightShape.GetDim(1)) / hiddenDivisor);

    context->SetBlockDim(SelectBlockDim(batch, 32U));
    StoreTilingData(context, tiling);
    return ge::GRAPH_SUCCESS;
}

}  // namespace timeseries_host

#endif  // TIME_SERIES_FORECAST_OP_HOST_UTILS_H
