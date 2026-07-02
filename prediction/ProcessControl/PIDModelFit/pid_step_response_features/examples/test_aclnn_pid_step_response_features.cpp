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
#include "pid_step_response_features_host.h"

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
    const int64_t candidates = 1;
    const int64_t sample_count = 6;
    const float sample_interval = 1.0f;
    const float settle_band_ratio = 0.05f;
    const std::vector<float> pv = {0.0f, 2.0f, 5.0f, 9.0f, 11.0f, 10.0f};
    const std::vector<float> sp = {0.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    std::vector<float> features(static_cast<size_t>(batch * candidates * PID_STEP_RESPONSE_FEATURE_COUNT), 0.0f);

    aclInit(nullptr);
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* d_pv = MallocDevice(pv.size() * sizeof(float));
    void* d_sp = MallocDevice(sp.size() * sizeof(float));
    void* d_features = MallocDevice(features.size() * sizeof(float));
    const uint64_t workspace_size = aclnnPidStepResponseFeaturesGetWorkspaceSize(batch, candidates, sample_count);
    void* workspace = MallocDevice(workspace_size);

    CHECK_ACL(aclrtMemcpy(d_pv, pv.size() * sizeof(float), pv.data(), pv.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_sp, sp.size() * sizeof(float), sp.data(), sp.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    const int32_t ret = aclnnPidStepResponseFeatures(
        d_pv, d_sp, d_features, batch, candidates, sample_count, sample_interval, settle_band_ratio, workspace,
        workspace_size, stream);
    CHECK_ACL(static_cast<aclError>(ret));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(features.data(), features.size() * sizeof(float), d_features,
                          features.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    const bool passed = std::fabs(features[0] - 0.0f) < 1.0e-3f && std::fabs(features[1] - 10.0f) < 1.0e-3f &&
                        std::fabs(features[2] - 0.0f) < 1.0e-3f && std::fabs(features[3] - 11.0f) < 1.0e-3f &&
                        std::fabs(features[5] - 0.1f) < 1.0e-3f && std::fabs(features[7] - 2.0f) < 1.0e-3f &&
                        std::fabs(features[8] - 4.0f) < 1.0e-3f && std::fabs(features[9] - 5.0f) < 1.0e-3f &&
                        std::fabs(features[10] - 15.0f) < 1.0e-3f && std::fabs(features[11] - 91.0f) < 1.0e-3f;

    std::cout << "PidStepResponseFeatures smoke initial=" << features[0] << " final=" << features[1]
              << " peak=" << features[3] << " overshoot=" << features[5] << " rise_time=" << features[7]
              << " peak_time=" << features[8] << " settling_time=" << features[9] << " iae=" << features[10]
              << " ise=" << features[11] << std::endl;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    aclrtFree(workspace);
    aclrtFree(d_features);
    aclrtFree(d_sp);
    aclrtFree(d_pv);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return passed ? 0 : 2;
}
