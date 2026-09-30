/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef EDGEPOOL_OP_IO_DEF_H
#define EDGEPOOL_OP_IO_DEF_H

#include "register/op_def_registry.h"

inline void RegisterEdgepoolIO(ops::OpDef* op)
{
    op->Input("order").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    op->Input("edge_index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    op->Input("score").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    op->Output("cluster").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    op->Output("selected_edge").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    op->Output("cluster_score").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    op->Output("counts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
}

#endif  // EDGEPOOL_OP_IO_DEF_H
