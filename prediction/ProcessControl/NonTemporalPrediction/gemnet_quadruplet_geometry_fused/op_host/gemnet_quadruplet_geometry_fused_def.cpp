/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "register/op_def_registry.h"

namespace ops
{
class GemNetQuadrupletGeometryFused : public OpDef
{
   public:
    explicit GemNetQuadrupletGeometryFused(const char* name) : OpDef(name)
    {
        this->Input("position").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        for (const char* input :
             {"source_index", "target_index", "interaction_source", "interaction_target", "reduce_ca", "expand_db",
              "reduce_intermediate_ca", "expand_intermediate_db", "reduce_intermediate_ab", "expand_intermediate_ab"})
        {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        }
        for (const char* output : {"angle_cab", "angle_abd", "theta_cabd"})
        {
            this->Output(output).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        }
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(GemNetQuadrupletGeometryFused);
}  // namespace ops
