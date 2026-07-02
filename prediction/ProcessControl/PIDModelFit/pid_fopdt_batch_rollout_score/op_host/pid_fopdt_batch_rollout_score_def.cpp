/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

using namespace ge;

namespace ops {
class PidFopdtBatchRolloutScore : public OpDef {
public:
    explicit PidFopdtBatchRolloutScore(const char* name) : OpDef(name)
    {
        this->Input("a").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("b").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("delay").ParamType(REQUIRED).DataType({DT_INT32}).Format({FORMAT_ND});
        this->Input("y0").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("sp").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("kp").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("ki").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("kd").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Output("best_result").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Output("best_idx").ParamType(REQUIRED).DataType({DT_INT32}).Format({FORMAT_ND});
        this->Attr("batch").AttrType(REQUIRED).Int();
        this->Attr("candidates").AttrType(REQUIRED).Int();
        this->Attr("sim_steps").AttrType(REQUIRED).Int();
        this->Attr("candidate_tile").AttrType(OPTIONAL).Int(64);
        this->Attr("sample_interval").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("settle_band").AttrType(OPTIONAL).Float(0.02f);
        this->Attr("overshoot_weight").AttrType(OPTIONAL).Float(50.0f);
        this->Attr("settling_weight").AttrType(OPTIONAL).Float(0.02f);
        this->Attr("control_weight").AttrType(OPTIONAL).Float(0.001f);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(PidFopdtBatchRolloutScore);
}  // namespace ops
