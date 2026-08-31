/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

extern "C" uint64_t aclnnCsrTagcnBasisK3FusedGetWorkspaceSize(int64_t nodes, int64_t edges, int64_t channels);
extern "C" int32_t aclnnCsrTagcnBasisK3Fused(void* row_ptr, void* source_index, void* norm, void* features, void* basis,
                                             int64_t nodes, int64_t edges, int64_t channels, void* workspace,
                                             uint64_t workspace_size, void* stream);
