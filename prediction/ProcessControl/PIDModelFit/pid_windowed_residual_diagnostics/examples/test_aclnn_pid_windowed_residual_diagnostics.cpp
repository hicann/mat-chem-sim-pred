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
#include "pid_windowed_residual_diagnostics_host.h"

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
    const int64_t batch = 1;
    const int64_t sample_count = 6;
    const int64_t window_size = 4;
    const int64_t stride = 2;
    const int64_t max_lag = 2;
    const int64_t window_count =
        aclnnPidWindowedResidualDiagnosticsGetWindowCount(sample_count, window_size, stride);
    const std::vector<float> actual = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> predicted = {1.0f, 1.0f, 3.0f, 5.0f, 5.0f, 4.0f};
    std::vector<float> metrics(
        static_cast<size_t>(batch * window_count * PID_WINDOWED_RESIDUAL_DIAGNOSTICS_METRIC_COUNT), 0.0f);
    std::vector<float> autocorr(static_cast<size_t>(batch * window_count * max_lag), 0.0f);

    aclInit(nullptr);
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* d_actual = MallocDevice(actual.size() * sizeof(float));
    void* d_predicted = MallocDevice(predicted.size() * sizeof(float));
    void* d_metrics = MallocDevice(metrics.size() * sizeof(float));
    void* d_autocorr = MallocDevice(autocorr.size() * sizeof(float));
    const uint64_t workspace_size =
        aclnnPidWindowedResidualDiagnosticsGetWorkspaceSize(batch, sample_count, window_size, stride, max_lag);
    void* workspace = MallocDevice(workspace_size);

    CHECK_ACL(aclrtMemcpy(d_actual, actual.size() * sizeof(float), actual.data(), actual.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_predicted, predicted.size() * sizeof(float), predicted.data(),
                          predicted.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    const int32_t ret = aclnnPidWindowedResidualDiagnostics(
        d_actual, d_predicted, d_metrics, d_autocorr, batch, sample_count, window_size, stride, max_lag, workspace,
        workspace_size, stream);
    CHECK_ACL(static_cast<aclError>(ret));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(metrics.data(), metrics.size() * sizeof(float), d_metrics, metrics.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(autocorr.data(), autocorr.size() * sizeof(float), d_autocorr,
                          autocorr.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    const bool passed =
        window_count == 2 && std::fabs(metrics[0] - 0.0f) < 1.0e-3f &&
        std::fabs(metrics[2] - 0.5f) < 1.0e-3f && std::fabs(metrics[3] - 0.70710678f) < 1.0e-3f &&
        std::fabs(metrics[4] - 1.0f) < 1.0e-3f && std::fabs(metrics[6] - 1.5f) < 1.0e-3f &&
        std::fabs(autocorr[0]) < 1.0e-3f && std::fabs(autocorr[1] + 0.5f) < 1.0e-3f &&
        std::isfinite(metrics[PID_WINDOWED_RESIDUAL_DIAGNOSTICS_METRIC_COUNT + 3]);

    std::cout << "PidWindowedResidualDiagnostics smoke windows=" << window_count << " w0_mean=" << metrics[0]
              << " w0_mae=" << metrics[2] << " w0_rmse=" << metrics[3] << " w0_dw=" << metrics[6]
              << " w0_autocorr=[" << autocorr[0] << ", " << autocorr[1] << "]" << std::endl;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    aclrtFree(workspace);
    aclrtFree(d_autocorr);
    aclrtFree(d_metrics);
    aclrtFree(d_predicted);
    aclrtFree(d_actual);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return passed ? 0 : 2;
}
