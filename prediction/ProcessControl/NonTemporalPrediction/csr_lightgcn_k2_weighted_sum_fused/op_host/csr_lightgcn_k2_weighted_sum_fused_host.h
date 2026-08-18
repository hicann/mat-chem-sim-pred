/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

extern "C" uint64_t aclnnCsrLightgcnK2WeightedSumFusedGetWorkspaceSize(int64_t nodes, int64_t edges, int64_t channels);
extern "C" int32_t aclnnCsrLightgcnK2WeightedSumFused(void* row_ptr, void* source_index, void* norm, void* features,
                                                      void* output, int64_t nodes, int64_t edges, int64_t channels,
                                                      float alpha0, float alpha1, float alpha2, void* workspace,
                                                      uint64_t workspace_size, void* stream);
