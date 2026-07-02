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
class PidStepResponseFeatures : public OpDef {
public:
    explicit PidStepResponseFeatures(const char* name) : OpDef(name)
    {
        this->Input("pv_candidates").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Input("sp").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Output("features").ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND});
        this->Attr("batch").AttrType(REQUIRED).Int();
        this->Attr("candidates").AttrType(REQUIRED).Int();
        this->Attr("sample_count").AttrType(REQUIRED).Int();
        this->Attr("sample_interval").AttrType(REQUIRED).Float();
        this->Attr("settle_band_ratio").AttrType(REQUIRED).Float();
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(PidStepResponseFeatures);
}  // namespace ops
