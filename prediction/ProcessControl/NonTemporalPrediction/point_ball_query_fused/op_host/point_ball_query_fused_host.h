/** Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0. */
#ifndef POINT_BALL_QUERY_FUSED_HOST_H
#define POINT_BALL_QUERY_FUSED_HOST_H
#include <cstdint>
extern "C" uint64_t aclnnPointBallQueryFusedGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t);
extern "C" int32_t aclnnPointBallQueryFused(void* points, void* queries, void* indices, void* counts, int64_t batch,
                                            int64_t point_count, int64_t query_count, int64_t sample_count,
                                            float radius, void* workspace, uint64_t workspace_size, void* stream);
#endif
