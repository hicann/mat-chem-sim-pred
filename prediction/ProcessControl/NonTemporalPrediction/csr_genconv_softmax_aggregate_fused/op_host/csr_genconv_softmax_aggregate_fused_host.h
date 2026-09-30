/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <cstdint>

extern "C" uint64_t aclnnCsrGenConvSoftmaxAggregateFusedGetWorkspaceSize(int64_t nodes, int64_t edges, int64_t channels,
                                                                         float temperature);
extern "C" int32_t aclnnCsrGenConvSoftmaxAggregateFused(void* row_ptr, void* source_index, void* features, void* output,
                                                        int64_t nodes, int64_t edges, int64_t channels,
                                                        float temperature, void* workspace, uint64_t workspace_size,
                                                        void* stream);
