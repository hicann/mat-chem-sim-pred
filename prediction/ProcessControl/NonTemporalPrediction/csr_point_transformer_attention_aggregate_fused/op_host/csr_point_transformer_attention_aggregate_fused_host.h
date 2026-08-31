/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

extern "C" uint64_t aclnnCsrPointTransformerAttentionAggregateFusedGetWorkspaceSize(int64_t nodes, int64_t edges,
                                                                                    int64_t channels,
                                                                                    int64_t max_segment_size);
extern "C" int32_t aclnnCsrPointTransformerAttentionAggregateFused(
    void* row_ptr, void* source_index, void* alpha_source, void* alpha_target, void* value, void* delta, void* output,
    int64_t nodes, int64_t edges, int64_t channels, int64_t max_segment_size, int64_t dtype, void* workspace,
    uint64_t workspace_size, void* stream);
