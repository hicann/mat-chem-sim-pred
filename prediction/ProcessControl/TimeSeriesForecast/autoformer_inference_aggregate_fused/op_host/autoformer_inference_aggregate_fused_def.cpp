/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../common/op_def_utils.h"

namespace optiling
{
ge::graphStatus AutoformerInferenceAggregateFusedTiling(gert::TilingContext* context);
}

namespace ge
{
graphStatus InferAutoformerInferenceAggregateFusedShape(gert::InferShapeContext* context);
graphStatus InferAutoformerInferenceAggregateFusedDataType(gert::InferDataTypeContext* context);
}  // namespace ge

namespace ops
{
class AutoformerInferenceAggregateFused : public TimeSeriesOpDef
{
   public:
    explicit AutoformerInferenceAggregateFused(const char* name) : TimeSeriesOpDef(name)
    {
        AddRequiredFloatNdInput("values");
        AddRequiredFloatNdInput("correlation");
        AddRequiredFloatNdOutput("output");
        this->Attr("top_k").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferAutoformerInferenceAggregateFusedShape)
            .SetInferDataType(ge::InferAutoformerInferenceAggregateFusedDataType);
        this->AICore().SetTiling(optiling::AutoformerInferenceAggregateFusedTiling);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AutoformerInferenceAggregateFused);
}  // namespace ops
