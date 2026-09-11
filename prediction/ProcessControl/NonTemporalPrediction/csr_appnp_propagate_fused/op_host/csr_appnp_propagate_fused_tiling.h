/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <cstdint>

struct CsrAppnpPropagateFusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t coreNum;
    float alpha;
    uint32_t reserved[3];
};
