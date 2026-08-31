/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_def_registry.h"

namespace ops
{
class DimeNetTripletEnumerateFused : public OpDef
{
   public:
    explicit DimeNetTripletEnumerateFused(const char* name) : OpDef(name)
    {
        this->Input("row_ptr").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("source_index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("idx_i").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("idx_j").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("idx_k").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("idx_kj").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("idx_ji").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("counts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("capacity").AttrType(REQUIRED).Int();
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(DimeNetTripletEnumerateFused);
}  // namespace ops
