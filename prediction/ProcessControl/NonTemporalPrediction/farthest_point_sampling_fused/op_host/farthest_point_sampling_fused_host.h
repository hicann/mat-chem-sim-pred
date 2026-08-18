/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef FARTHEST_POINT_SAMPLING_FUSED_HOST_H
#define FARTHEST_POINT_SAMPLING_FUSED_HOST_H

#include <cstdint>

extern "C" int32_t aclnnFarthestPointSamplingFused(void* points, void* sample_indices, int64_t batch,
                                                   int64_t point_count, int64_t sample_count, void* workspace,
                                                   uint64_t workspace_size, void* stream);
extern "C" uint64_t aclnnFarthestPointSamplingFusedGetWorkspaceSize(int64_t batch, int64_t point_count,
                                                                    int64_t sample_count);

#endif  // FARTHEST_POINT_SAMPLING_FUSED_HOST_H
