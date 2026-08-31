/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#pragma once

#include <cstdint>

extern "C" uint64_t aclnnPpfPointPairFeaturesFusedGetWorkspaceSize(int64_t nodes, int64_t edges);
extern "C" int32_t aclnnPpfPointPairFeaturesFused(void* position, void* normal, void* sourceIndex, void* targetIndex,
                                                  void* output, int64_t nodes, int64_t edges, void* workspace,
                                                  uint64_t workspaceSize, void* stream);
