/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include "register/op_def_registry.h"

namespace ops
{

class FarthestPointSamplingFused : public OpDef
{
   public:
    explicit FarthestPointSamplingFused(const char* name) : OpDef(name)
    {
        this->Input("points").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("sample_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("sample_count").AttrType(REQUIRED).Int();
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(FarthestPointSamplingFused);

}  // namespace ops
