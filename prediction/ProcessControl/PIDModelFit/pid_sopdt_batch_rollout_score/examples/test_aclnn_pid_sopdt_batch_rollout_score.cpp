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
#include "pid_sopdt_batch_rollout_score_host.h"

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
    const int64_t sim_steps = argc > 2 ? std::atoll(argv[2]) : 64;
    const int64_t candidates = argc > 3 ? std::atoll(argv[3]) : 3;
    const int64_t candidate_tile = argc > 4 ? std::atoll(argv[4]) : 2;
    const float sample_interval = 1.0f;
    const float settle_band = 0.05f;
    const float overshoot_weight = 50.0f;
    const float settling_weight = 0.02f;
    const float control_weight = 0.001f;
    const std::vector<float> a1 = {1.7f};
    const std::vector<float> a2 = {-0.72f};
    const std::vector<float> b = {0.02f};
    const std::vector<int32_t> delay = {2};
    const std::vector<float> y0 = {0.0f};
    const std::vector<float> sp = {1.0f};
    std::vector<float> kp(static_cast<size_t>(candidates), 0.0f);
    std::vector<float> ki(static_cast<size_t>(candidates), 0.0f);
    std::vector<float> kd(static_cast<size_t>(candidates), 0.0f);
    for (int64_t i = 0; i < candidates; ++i) {
        const float ratio = static_cast<float>(i) / static_cast<float>(std::max<int64_t>(1, candidates - 1));
        kp[static_cast<size_t>(i)] = 0.05f + 0.75f * ratio;
        ki[static_cast<size_t>(i)] = 0.08f * ratio;
        kd[static_cast<size_t>(i)] = 0.0f;
    }
    std::vector<float> best_result(static_cast<size_t>(batch * PID_SOPDT_BATCH_ROLLOUT_RESULT_COUNT), 0.0f);
    std::vector<int32_t> best_idx(static_cast<size_t>(batch), -1);

    aclInit(nullptr);
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* d_a1 = MallocDevice(a1.size() * sizeof(float));
    void* d_a2 = MallocDevice(a2.size() * sizeof(float));
    void* d_b = MallocDevice(b.size() * sizeof(float));
    void* d_delay = MallocDevice(delay.size() * sizeof(int32_t));
    void* d_y0 = MallocDevice(y0.size() * sizeof(float));
    void* d_sp = MallocDevice(sp.size() * sizeof(float));
    void* d_kp = MallocDevice(kp.size() * sizeof(float));
    void* d_ki = MallocDevice(ki.size() * sizeof(float));
    void* d_kd = MallocDevice(kd.size() * sizeof(float));
    void* d_best_result = MallocDevice(best_result.size() * sizeof(float));
    void* d_best_idx = MallocDevice(best_idx.size() * sizeof(int32_t));
    const uint64_t workspace_size =
        aclnnPidSopdtBatchRolloutScoreGetWorkspaceSize(batch, candidates, sim_steps, candidate_tile);
    void* workspace = MallocDevice(workspace_size);

    CHECK_ACL(aclrtMemcpy(d_a1, a1.size() * sizeof(float), a1.data(), a1.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_a2, a2.size() * sizeof(float), a2.data(), a2.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_b, b.size() * sizeof(float), b.data(), b.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_delay, delay.size() * sizeof(int32_t), delay.data(), delay.size() * sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_y0, y0.size() * sizeof(float), y0.data(), y0.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_sp, sp.size() * sizeof(float), sp.data(), sp.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_kp, kp.size() * sizeof(float), kp.data(), kp.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_ki, ki.size() * sizeof(float), ki.data(), ki.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_kd, kd.size() * sizeof(float), kd.data(), kd.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    const int32_t ret = aclnnPidSopdtBatchRolloutScore(
        d_a1, d_a2, d_b, d_delay, d_y0, d_sp, d_kp, d_ki, d_kd, d_best_result, d_best_idx, batch, candidates, sim_steps,
        candidate_tile, sample_interval, settle_band, overshoot_weight, settling_weight, control_weight, workspace,
        workspace_size, stream);
    CHECK_ACL(static_cast<aclError>(ret));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(best_result.data(), best_result.size() * sizeof(float), d_best_result,
                          best_result.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(best_idx.data(), best_idx.size() * sizeof(int32_t), d_best_idx,
                          best_idx.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));

    const bool passed = best_idx[0] > 0 && std::isfinite(best_result[0]) && best_result[0] > 0.0f &&
                        best_result[4] > 0.0f && best_result[5] > 0.0f;
    std::cout << "PidSopdtBatchRolloutScore smoke best_idx=" << best_idx[0] << " best_score=" << best_result[0]
              << " best_kp=" << best_result[1] << " best_ki=" << best_result[2]
              << " best_iae=" << best_result[4] << " best_settling=" << best_result[7] << std::endl;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    aclrtFree(workspace);
    aclrtFree(d_best_idx);
    aclrtFree(d_best_result);
    aclrtFree(d_kd);
    aclrtFree(d_ki);
    aclrtFree(d_kp);
    aclrtFree(d_sp);
    aclrtFree(d_y0);
    aclrtFree(d_delay);
    aclrtFree(d_b);
    aclrtFree(d_a2);
    aclrtFree(d_a1);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return passed ? 0 : 2;
}
