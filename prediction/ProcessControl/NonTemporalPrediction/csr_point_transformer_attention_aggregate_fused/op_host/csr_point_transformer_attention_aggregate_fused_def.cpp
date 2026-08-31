/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_def_registry.h"

namespace ops
{
class CsrPointTransformerAttentionAggregateFused : public OpDef
{
   public:
    explicit CsrPointTransformerAttentionAggregateFused(const char* name) : OpDef(name)
    {
        this->Input("row_ptr").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("source_index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        const std::initializer_list<ge::DataType> floatingTypes{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
        this->Input("alpha_source").ParamType(REQUIRED).DataType(floatingTypes).Format({ge::FORMAT_ND});
        this->Input("alpha_target").ParamType(REQUIRED).DataType(floatingTypes).Format({ge::FORMAT_ND});
        this->Input("value").ParamType(REQUIRED).DataType(floatingTypes).Format({ge::FORMAT_ND});
        this->Input("delta").ParamType(REQUIRED).DataType(floatingTypes).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType(floatingTypes).Format({ge::FORMAT_ND});
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(CsrPointTransformerAttentionAggregateFused);
}  // namespace ops
