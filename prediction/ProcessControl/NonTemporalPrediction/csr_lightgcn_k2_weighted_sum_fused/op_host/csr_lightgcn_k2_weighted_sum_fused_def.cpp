/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_def_registry.h"

namespace ops
{
class CsrLightgcnK2WeightedSumFused : public OpDef
{
   public:
    explicit CsrLightgcnK2WeightedSumFused(const char* name) : OpDef(name)
    {
        this->Input("row_ptr").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("source_index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("norm").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("features").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("alpha0").AttrType(REQUIRED).Float();
        this->Attr("alpha1").AttrType(REQUIRED).Float();
        this->Attr("alpha2").AttrType(REQUIRED).Float();
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(CsrLightgcnK2WeightedSumFused);
}  // namespace ops
