/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

extern "C" uint64_t aclnnCsrHypergraphAttentionTwoStagePropagateFusedGetWorkspaceSize(int64_t nodes, int64_t hyperedges,
                                                                                      int64_t incidences, int64_t heads,
                                                                                      int64_t channels,
                                                                                      int64_t max_edge_size,
                                                                                      int64_t max_node_degree);
extern "C" int32_t aclnnCsrHypergraphAttentionTwoStagePropagateFused(
    void* edge_row_ptr, void* node_index, void* edge_scale, void* node_row_ptr, void* edge_index,
    void* incidence_position, void* node_scale, void* features, void* attention_logits, void* output, int64_t nodes,
    int64_t hyperedges, int64_t incidences, int64_t heads, int64_t channels, int64_t max_edge_size,
    int64_t max_node_degree, int64_t dtype, float negative_slope, void* workspace, uint64_t workspace_size,
    void* stream);
