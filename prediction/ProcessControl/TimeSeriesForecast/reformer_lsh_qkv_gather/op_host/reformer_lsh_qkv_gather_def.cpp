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
ge::graphStatus ReformerLshQkvGatherTiling(gert::TilingContext* context);
}

namespace ge
{
graphStatus InferReformerLshQkvGatherShape(gert::InferShapeContext* context);
graphStatus InferReformerLshQkvGatherDataType(gert::InferDataTypeContext* context);
}  // namespace ge

namespace ops
{
class ReformerLshQkvGather : public TimeSeriesOpDef
{
   public:
    explicit ReformerLshQkvGather(const char* name) : TimeSeriesOpDef(name)
    {
        AddRequiredFloatNdInput("query_key");
        AddRequiredFloatNdInput("value");
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        AddRequiredFloatNdOutput("sorted_query_key");
        AddRequiredFloatNdOutput("sorted_value");
        this->SetInferShape(ge::InferReformerLshQkvGatherShape).SetInferDataType(ge::InferReformerLshQkvGatherDataType);
        this->AICore().SetTiling(optiling::ReformerLshQkvGatherTiling);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ReformerLshQkvGather);
}  // namespace ops
