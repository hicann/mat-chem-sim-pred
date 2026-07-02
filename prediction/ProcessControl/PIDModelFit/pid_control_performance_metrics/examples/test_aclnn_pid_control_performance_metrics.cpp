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
#include <cstdlib>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "pid_control_performance_metrics_host.h"

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
    const float sample_interval = 1.0f;
    const float settle_band = 0.1f;
    const std::vector<float> pv = {
        50.0f, 51.0f, 52.0f, 50.0f,
        48.0f, 49.0f, 50.0f, 51.0f,
    };
    const std::vector<float> sp = {
        50.0f, 50.0f, 50.0f, 50.0f,
        50.0f, 50.0f, 50.0f, 50.0f,
    };
    const std::vector<float> lsl = {47.0f, 47.0f};
    const std::vector<float> usl = {53.0f, 53.0f};
    const std::vector<float> mv_variance = {0.25f, 0.25f};
    std::vector<float> metrics(static_cast<size_t>(batch * PID_CONTROL_PERFORMANCE_METRIC_COUNT), 0.0f);

    aclInit(nullptr);
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* d_pv = MallocDevice(pv.size() * sizeof(float));
    void* d_sp = MallocDevice(sp.size() * sizeof(float));
    void* d_lsl = MallocDevice(lsl.size() * sizeof(float));
    void* d_usl = MallocDevice(usl.size() * sizeof(float));
    void* d_mv_variance = MallocDevice(mv_variance.size() * sizeof(float));
    void* d_metrics = MallocDevice(metrics.size() * sizeof(float));
    const uint64_t workspace_size = aclnnPidControlPerformanceMetricsGetWorkspaceSize(batch, sample_count);
    void* workspace = MallocDevice(workspace_size);

    CHECK_ACL(aclrtMemcpy(d_pv, pv.size() * sizeof(float), pv.data(), pv.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_sp, sp.size() * sizeof(float), sp.data(), sp.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_lsl, lsl.size() * sizeof(float), lsl.data(), lsl.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_usl, usl.size() * sizeof(float), usl.data(), usl.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_mv_variance, mv_variance.size() * sizeof(float), mv_variance.data(),
                          mv_variance.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    const int32_t ret = aclnnPidControlPerformanceMetrics(
        d_pv, d_sp, d_lsl, d_usl, d_mv_variance, d_metrics, batch, sample_count, sample_interval, settle_band,
        workspace, workspace_size, stream);
    CHECK_ACL(static_cast<aclError>(ret));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(metrics.data(), metrics.size() * sizeof(float), d_metrics, metrics.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    const float mean0 = metrics[0];
    const float iae0 = metrics[8];
    const float ise0 = metrics[9];
    const float settling0 = metrics[18];
    const float out_count1 = metrics[PID_CONTROL_PERFORMANCE_METRIC_COUNT + 15];
    const bool passed = std::fabs(mean0 - 50.75f) < 1.0e-3f && std::fabs(iae0 - 3.0f) < 1.0e-3f &&
                        std::fabs(ise0 - 5.0f) < 1.0e-3f && std::fabs(settling0 - 3.0f) < 1.0e-3f &&
                        std::fabs(out_count1 - 0.0f) < 1.0e-3f;

    std::cout << "PidControlPerformanceMetrics smoke mean0=" << mean0 << " iae0=" << iae0
              << " ise0=" << ise0 << " settling0=" << settling0 << " out_count1=" << out_count1 << std::endl;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    aclrtFree(workspace);
    aclrtFree(d_metrics);
    aclrtFree(d_mv_variance);
    aclrtFree(d_usl);
    aclrtFree(d_lsl);
    aclrtFree(d_sp);
    aclrtFree(d_pv);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return passed ? 0 : 2;
}
