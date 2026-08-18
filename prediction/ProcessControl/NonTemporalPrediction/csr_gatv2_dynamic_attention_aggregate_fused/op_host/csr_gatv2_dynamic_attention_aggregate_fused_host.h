/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

extern "C" uint64_t aclnnCsrGatv2DynamicAttentionAggregateFusedGetWorkspaceSize(int64_t nodes, int64_t edges,
                                                                                int64_t heads, int64_t channels,
                                                                                int64_t max_segment_size);
extern "C" int32_t aclnnCsrGatv2DynamicAttentionAggregateFused(void* row_ptr, void* source_index,
                                                               void* source_projected, void* target_projected,
                                                               void* attention, void* output, int64_t nodes,
                                                               int64_t edges, int64_t heads, int64_t channels,
                                                               int64_t max_segment_size, float negative_slope,
                                                               void* workspace, uint64_t workspace_size, void* stream);
