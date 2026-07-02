/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "pid_tuning_rule_batch_host.h"

namespace {

void CheckAcl(aclError ret, const char* expr)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(expr) + " failed, ret=" + std::to_string(ret));
    }
}

#define CHECK_ACL(expr) CheckAcl((expr), #expr)

bool Near(float lhs, float rhs, float tol = 1.0e-4f)
{
    return std::fabs(lhs - rhs) <= tol;
}

}  // namespace

int main(int argc, char** argv)
{
    const int32_t device_id = argc > 1 ? static_cast<int32_t>(std::strtol(argv[1], nullptr, 10)) : 0;
    constexpr int64_t batch = 8;
    std::vector<float> gain = {2.0f, 1.0f, 0.0f, 1.5f, 2.5f, 1.2f, 0.8f, 3.0f};
    std::vector<float> tau = {10.0f, 20.0f, 10.0f, 0.0f, 40.0f, 30.0f, 18.0f, 12.0f};
    std::vector<float> theta = {2.0f, 5.0f, 2.0f, 4.0f, 8.0f, 6.0f, 3.0f, 2.5f};
    std::vector<float> lambda = {4.0f, 10.0f, 4.0f, 8.0f, 24.0f, 12.0f, 6.0f, 5.0f};
    std::vector<float> pid(
        static_cast<size_t>(batch * PID_TUNING_RULE_BATCH_RULE_COUNT * PID_TUNING_RULE_BATCH_PARAM_COUNT), 0.0f);
    std::vector<float> diag(
        static_cast<size_t>(batch * PID_TUNING_RULE_BATCH_RULE_COUNT * PID_TUNING_RULE_BATCH_DIAGNOSTIC_COUNT), 0.0f);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* d_gain = nullptr;
    void* d_tau = nullptr;
    void* d_theta = nullptr;
    void* d_lambda = nullptr;
    void* d_pid = nullptr;
    void* d_diag = nullptr;
    void* workspace = nullptr;
    const size_t input_bytes = gain.size() * sizeof(float);
    const size_t pid_bytes = pid.size() * sizeof(float);
    const size_t diag_bytes = diag.size() * sizeof(float);
    const uint64_t workspace_size = aclnnPidTuningRuleBatchGetWorkspaceSize(batch);

    CHECK_ACL(aclrtMalloc(&d_gain, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_tau, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_theta, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_lambda, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_pid, pid_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_diag, diag_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_gain, input_bytes, gain.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_tau, input_bytes, tau.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_theta, input_bytes, theta.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_lambda, input_bytes, lambda.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    const int32_t ret =
        aclnnPidTuningRuleBatch(d_gain, d_tau, d_theta, d_lambda, d_pid, d_diag, batch, workspace, workspace_size, stream);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclnnPidTuningRuleBatch failed, ret=" + std::to_string(ret));
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(pid.data(), pid_bytes, d_pid, pid_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(diag.data(), diag_bytes, d_diag, diag_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    if (!Near(pid[0], 3.0f) || !Near(pid[1], 0.75f) || !Near(pid[2], 3.0f)) {
        std::cerr << "ZN expected Kp/Ki/Kd=3/0.75/3, got " << pid[0] << ", " << pid[1] << ", " << pid[2]
                  << std::endl;
        return 1;
    }
    if (!Near(diag[0], 1.0f) || !Near(diag[1], 0.2f)) {
        std::cerr << "warning: unexpected diagnostics[0:2]: " << diag[0] << ", " << diag[1] << std::endl;
    }
    const size_t invalid_base = static_cast<size_t>(2 * PID_TUNING_RULE_BATCH_RULE_COUNT *
                                                    PID_TUNING_RULE_BATCH_DIAGNOSTIC_COUNT);
    if (!Near(diag[invalid_base], 0.0f)) {
        std::cerr << "warning: invalid diagnostic row was not filtered" << std::endl;
    }

    CHECK_ACL(aclrtFree(workspace));
    CHECK_ACL(aclrtFree(d_diag));
    CHECK_ACL(aclrtFree(d_pid));
    CHECK_ACL(aclrtFree(d_lambda));
    CHECK_ACL(aclrtFree(d_theta));
    CHECK_ACL(aclrtFree(d_tau));
    CHECK_ACL(aclrtFree(d_gain));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(device_id));
    CHECK_ACL(aclFinalize());
    std::cout << "PidTuningRuleBatch smoke passed." << std::endl;
    return 0;
}
