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
class PidTuningRuleBatch : public OpDef {
public:
    explicit PidTuningRuleBatch(const char* name) : OpDef(name)
    {
        this->Input("process_gain").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("time_constant").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("dead_time").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("lambda_value").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Output("pid_params").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Output("diagnostics").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Attr("batch").AttrType(REQUIRED).Int();
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(PidTuningRuleBatch);
}  // namespace ops
