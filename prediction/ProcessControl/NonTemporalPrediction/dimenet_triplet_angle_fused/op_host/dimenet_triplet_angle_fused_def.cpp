/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "register/op_def_registry.h"

namespace ops
{
class DimeNetTripletAngleFused : public OpDef
{
   public:
    explicit DimeNetTripletAngleFused(const char* name) : OpDef(name)
    {
        this->Input("position").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        for (const char* input : {"idx_i", "idx_j", "idx_k"})
        {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        }
        this->Output("angle").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(DimeNetTripletAngleFused);
}  // namespace ops
