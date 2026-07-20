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

// The callbacks are implemented separately from registration so generated Host
// scaffolding does not obscure the operator-specific interface definition.
namespace optiling {
ge::graphStatus UnicornnScanFusedTiling(gert::TilingContext* context);
}

namespace ge {
graphStatus InferUnicornnScanFusedShape(gert::InferShapeContext* context);
graphStatus InferUnicornnScanFusedDataType(gert::InferDataTypeContext* context);
}

namespace ops {
class UnicornnScanFused : public TimeSeriesOpDef {
public:
    explicit UnicornnScanFused(const char* name) : TimeSeriesOpDef(name)
    {
        AddRequiredFloatNdInput("x");
        AddRequiredFloatNdInput("weight");
        AddRequiredFloatNdInput("bias");
        AddRequiredFloatNdOutput("output");

        this->SetInferShape(ge::InferUnicornnScanFusedShape)
            .SetInferDataType(ge::InferUnicornnScanFusedDataType);
        this->AICore().SetTiling(optiling::UnicornnScanFusedTiling);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(UnicornnScanFused);
}  // namespace ops
