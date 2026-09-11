/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>
extern "C" uint64_t aclnnCsrSageMeanRootReluFusedGetWorkspaceSize(int64_t nodes, int64_t edges, int64_t channels);
extern "C" int32_t aclnnCsrSageMeanRootReluFused(void* row_ptr, void* column_index, void* neighbor_features,
                                                 void* root_features, void* output, int64_t nodes, int64_t edges,
                                                 int64_t channels, void* workspace, uint64_t workspace_size,
                                                 void* stream);
