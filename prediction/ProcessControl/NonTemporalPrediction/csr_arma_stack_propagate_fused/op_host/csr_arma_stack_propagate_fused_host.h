/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

extern "C" uint64_t aclnnCsrArmaStackPropagateFusedGetWorkspaceSize(int64_t nodes, int64_t edges, int64_t stacks,
                                                                    int64_t channels);
extern "C" int32_t aclnnCsrArmaStackPropagateFused(void* row_ptr, void* source_index, void* edge_weight,
                                                   void* projected, void* root, void* bias, void* output, int64_t nodes,
                                                   int64_t edges, int64_t stacks, int64_t channels, int64_t relu,
                                                   void* workspace, uint64_t workspace_size, void* stream);
