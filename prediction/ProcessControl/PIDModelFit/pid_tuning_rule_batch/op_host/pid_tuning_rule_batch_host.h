/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PID_TUNING_RULE_BATCH_HOST_H
#define PID_TUNING_RULE_BATCH_HOST_H

#include <cstdint>

constexpr int64_t PID_TUNING_RULE_BATCH_RULE_COUNT = 3;
constexpr int64_t PID_TUNING_RULE_BATCH_PARAM_COUNT = 3;
constexpr int64_t PID_TUNING_RULE_BATCH_DIAGNOSTIC_COUNT = 4;

extern "C" int32_t aclnnPidTuningRuleBatch(
    void* process_gain, void* time_constant, void* dead_time, void* lambda_value, void* pid_params,
    void* diagnostics, int64_t batch, void* workspace, uint64_t workspace_size, void* stream);

extern "C" uint64_t aclnnPidTuningRuleBatchGetWorkspaceSize(int64_t batch);

#endif  // PID_TUNING_RULE_BATCH_HOST_H
