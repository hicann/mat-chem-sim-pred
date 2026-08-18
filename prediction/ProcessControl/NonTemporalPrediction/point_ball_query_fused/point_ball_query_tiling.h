/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef POINT_BALL_QUERY_TILING_H
#define POINT_BALL_QUERY_TILING_H

#include <cstdint>

struct BallQueryTiling
{
    uint32_t batch;
    uint32_t pointCount;
    uint32_t queryCount;
    uint32_t sampleCount;
    uint32_t totalQueries;
    uint32_t coreNum;
    float radiusSquared;
};

#endif
