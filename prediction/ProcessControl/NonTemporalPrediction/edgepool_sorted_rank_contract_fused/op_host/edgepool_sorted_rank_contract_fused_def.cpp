/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "edgepool_op_io_def.h"

namespace ops
{
class EdgepoolSortedRankContractFused : public OpDef
{
   public:
    explicit EdgepoolSortedRankContractFused(const char* name) : OpDef(name)
    {
        RegisterEdgepoolIO(this);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(EdgepoolSortedRankContractFused);
}  // namespace ops
