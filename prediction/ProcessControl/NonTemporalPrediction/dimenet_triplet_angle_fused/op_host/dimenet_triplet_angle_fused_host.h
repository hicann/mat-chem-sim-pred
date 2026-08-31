/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#pragma once

#include <cstdint>

extern "C" uint64_t aclnnDimeNetTripletAngleFusedGetWorkspaceSize(int64_t nodes, int64_t triplets);
extern "C" int32_t aclnnDimeNetTripletAngleFused(void* position, void* idxI, void* idxJ, void* idxK, void* angle,
                                                 int64_t nodes, int64_t triplets, void* workspace,
                                                 uint64_t workspaceSize, void* stream);
