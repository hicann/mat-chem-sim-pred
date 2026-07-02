/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "pid_process_capability_metrics_host.h"

namespace {

void CheckAcl(aclError ret, const char* expr)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << expr << " failed, ret=" << ret << std::endl;
        std::exit(1);
    }
}

#define CHECK_ACL(expr) CheckAcl((expr), #expr)

void* MallocDevice(size_t bytes)
{
    void* ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

}  // namespace

int main(int argc, char** argv)
{
    const int device_id = argc > 1 ? std::atoi(argv[1]) : 0;
    const int64_t batch = 2;
    const int64_t sample_count = 4;
    const std::vector<float> values = {
        49.0f, 50.0f, 51.0f, 54.0f,
        48.0f, 49.0f, 50.0f, 51.0f,
    };
    const std::vector<float> lsl = {47.0f, 47.0f};
    const std::vector<float> usl = {53.0f, 53.0f};
    std::vector<float> metrics(static_cast<size_t>(batch * PID_PROCESS_CAPABILITY_METRIC_COUNT), 0.0f);

    aclInit(nullptr);
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* d_values = MallocDevice(values.size() * sizeof(float));
    void* d_lsl = MallocDevice(lsl.size() * sizeof(float));
    void* d_usl = MallocDevice(usl.size() * sizeof(float));
    void* d_metrics = MallocDevice(metrics.size() * sizeof(float));
    const uint64_t workspace_size = aclnnPidProcessCapabilityMetricsGetWorkspaceSize(batch, sample_count);
    void* workspace = MallocDevice(workspace_size);

    CHECK_ACL(aclrtMemcpy(d_values, values.size() * sizeof(float), values.data(), values.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_lsl, lsl.size() * sizeof(float), lsl.data(), lsl.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_usl, usl.size() * sizeof(float), usl.data(), usl.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    const int32_t ret = aclnnPidProcessCapabilityMetrics(
        d_values, d_lsl, d_usl, d_metrics, batch, sample_count, workspace, workspace_size, stream);
    CHECK_ACL(static_cast<aclError>(ret));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(metrics.data(), metrics.size() * sizeof(float), d_metrics, metrics.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    const float mean0 = metrics[0];
    const float out_count0 = metrics[10];
    const float min1 = metrics[PID_PROCESS_CAPABILITY_METRIC_COUNT + 11];
    const float max1 = metrics[PID_PROCESS_CAPABILITY_METRIC_COUNT + 12];
    const bool passed = std::fabs(mean0 - 51.0f) < 1.0e-3f && std::fabs(out_count0 - 1.0f) < 1.0e-3f &&
                        std::fabs(min1 - 48.0f) < 1.0e-3f && std::fabs(max1 - 51.0f) < 1.0e-3f;

    std::cout << "PidProcessCapabilityMetrics smoke mean0=" << mean0 << " cpk0=" << metrics[6]
              << " out_count0=" << out_count0 << " min1=" << min1 << " max1=" << max1 << std::endl;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    aclrtFree(workspace);
    aclrtFree(d_metrics);
    aclrtFree(d_usl);
    aclrtFree(d_lsl);
    aclrtFree(d_values);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return passed ? 0 : 2;
}
