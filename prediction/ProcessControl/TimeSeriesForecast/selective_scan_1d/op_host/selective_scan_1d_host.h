/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECTIVE_SCAN_1D_HOST_H
#define SELECTIVE_SCAN_1D_HOST_H

#include <cstdint>

extern "C" int32_t aclnnSelectiveScan1D(
    void* u, void* delta, void* a, void* b, void* c, void* d, void* output, int64_t batch, int64_t length,
    int64_t dim, int64_t state, void* workspace, uint64_t workspace_size, void* stream);

extern "C" uint64_t aclnnSelectiveScan1DGetWorkspaceSize(
    int64_t batch, int64_t length, int64_t dim, int64_t state);

#endif  // SELECTIVE_SCAN_1D_HOST_H
