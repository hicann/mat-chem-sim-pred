/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

extern "C" uint64_t aclnnCsrTransformerDotAttentionAggregateFusedGetWorkspaceSize(int64_t nodes, int64_t edges,
                                                                                  int64_t heads, int64_t channels,
                                                                                  int64_t max_segment_size);
extern "C" int32_t aclnnCsrTransformerDotAttentionAggregateFused(void* row_ptr, void* source_index, void* query,
                                                                 void* key, void* value, void* output, int64_t nodes,
                                                                 int64_t edges, int64_t heads, int64_t channels,
                                                                 int64_t max_segment_size, void* workspace,
                                                                 uint64_t workspace_size, void* stream);
