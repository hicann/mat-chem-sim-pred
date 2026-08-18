/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef ATTENTION_FUSED_TILING_H
#define ATTENTION_FUSED_TILING_H

#include <cstdint>

struct AttentionFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t heads;
    uint32_t channels;
    uint32_t coreNum;
    uint32_t maxSegmentSize;
    float parameter;
    uint32_t reserved;
};

#endif  // ATTENTION_FUSED_TILING_H
