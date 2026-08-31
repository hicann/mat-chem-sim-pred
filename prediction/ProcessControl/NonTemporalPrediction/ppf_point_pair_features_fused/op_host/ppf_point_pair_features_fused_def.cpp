/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "register/op_def_registry.h"

namespace ops
{
class PpfPointPairFeaturesFused : public OpDef
{
   public:
    explicit PpfPointPairFeaturesFused(const char* name) : OpDef(name)
    {
        this->Input("position").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("normal").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("source_index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("target_index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("features").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(PpfPointPairFeaturesFused);
}  // namespace ops
