/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

extern "C" uint64_t aclnnCsrSignedCrossMeanPackFusedGetWorkspaceSize(int64_t nodes, int64_t positive_edges,
                                                                     int64_t negative_edges, int64_t channels,
                                                                     int64_t max_segment_size);
extern "C" int32_t aclnnCsrSignedCrossMeanPackFused(void* positive_row_ptr, void* positive_source_index,
                                                    void* negative_row_ptr, void* negative_source_index, void* features,
                                                    void* positive_inverse_degree, void* negative_inverse_degree,
                                                    void* output, int64_t nodes, int64_t positive_edges,
                                                    int64_t negative_edges, int64_t channels, int64_t max_segment_size,
                                                    void* workspace, uint64_t workspace_size, void* stream);
