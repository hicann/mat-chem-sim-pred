/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

extern "C" uint64_t aclnnCsrGcn2ResidualPropagateFusedGetWorkspaceSize(int64_t nodes, int64_t edges, int64_t channels);
extern "C" int32_t aclnnCsrGcn2ResidualPropagateFused(void* row_ptr, void* source_index, void* edge_weight,
                                                      void* current, void* initial, void* output, int64_t nodes,
                                                      int64_t edges, int64_t channels, float alpha, void* workspace,
                                                      uint64_t workspace_size, void* stream);
