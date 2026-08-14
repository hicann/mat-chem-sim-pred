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
ge::graphStatus ReformerLshBucketSortTiling(gert::TilingContext* context);
}

namespace ge
{
graphStatus InferReformerLshBucketSortShape(gert::InferShapeContext* context);
graphStatus InferReformerLshBucketSortDataType(gert::InferDataTypeContext* context);
}  // namespace ge

namespace ops
{
class ReformerLshBucketSort : public TimeSeriesOpDef
{
   public:
    explicit ReformerLshBucketSort(const char* name) : TimeSeriesOpDef(name)
    {
        this->Input("keys")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sorted_keys")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sticker")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("inverse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("sequence_length").AttrType(REQUIRED).Int();
        this->Attr("total_buckets").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferReformerLshBucketSortShape)
            .SetInferDataType(ge::InferReformerLshBucketSortDataType);
        this->AICore().SetTiling(optiling::ReformerLshBucketSortTiling);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ReformerLshBucketSort);
}  // namespace ops
