/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_def_registry.h"

namespace ops
{
class CsrSignedCrossMeanPackFused : public OpDef
{
   public:
    explicit CsrSignedCrossMeanPackFused(const char* name) : OpDef(name)
    {
        this->Input("positive_row_ptr").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("positive_source_index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("negative_row_ptr").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("negative_source_index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("features").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("positive_inverse_degree").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("negative_inverse_degree").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(CsrSignedCrossMeanPackFused);
}  // namespace ops
