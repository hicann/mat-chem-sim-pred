/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

extern "C" uint64_t aclnnDimeNetTripletEnumerateFusedGetWorkspaceSize(int64_t nodes, int64_t edges, int64_t capacity);
extern "C" int32_t aclnnDimeNetTripletEnumerateFused(void* row_ptr, void* source_index, void* idx_i, void* idx_j,
                                                     void* idx_k, void* idx_kj, void* idx_ji, void* counts,
                                                     int64_t nodes, int64_t edges, int64_t capacity, void* workspace,
                                                     uint64_t workspace_size, void* stream);
