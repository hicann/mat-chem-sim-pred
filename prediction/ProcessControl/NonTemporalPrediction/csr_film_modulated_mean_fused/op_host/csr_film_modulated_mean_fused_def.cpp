/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_def_registry.h"

namespace ops
{
class CsrFilmModulatedMeanFused : public OpDef
{
   public:
    explicit CsrFilmModulatedMeanFused(const char* name) : OpDef(name)
    {
        this->Input("row_ptr").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("source_index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("projected").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("apply_relu").AttrType(REQUIRED).Bool();
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(CsrFilmModulatedMeanFused);
}  // namespace ops
