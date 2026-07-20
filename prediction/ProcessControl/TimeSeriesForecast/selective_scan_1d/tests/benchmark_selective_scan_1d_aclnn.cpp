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
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "selective_scan_1d_host.h"

namespace {

void CheckAcl(aclError ret, const char* expr)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << expr << " failed, ret=" << ret << std::endl;
        std::exit(1);
    }
}

#define CHECK_ACL(expr) CheckAcl((expr), #expr)

int64_t Arg(int argc, char** argv, int index, int64_t fallback)
{
    return argc > index ? std::atoll(argv[index]) : fallback;
}

void* MallocDevice(size_t bytes)
{
    void* ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

std::vector<float> RandomVector(size_t size, float low, float high, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> data(size);
    for (float& value : data) {
        value = dist(rng);
    }
    return data;
}

std::vector<float> CpuReference(
    const std::vector<float>& u, const std::vector<float>& delta, const std::vector<float>& a,
    const std::vector<float>& b, const std::vector<float>& c, const std::vector<float>& d, int64_t batch,
    int64_t length, int64_t dim, int64_t state)
{
    std::vector<float> output(static_cast<size_t>(batch * length * dim), 0.0f);
    std::vector<float> state_value(static_cast<size_t>(batch * dim * state), 0.0f);
    for (int64_t bi = 0; bi < batch; ++bi) {
        for (int64_t di = 0; di < dim; ++di) {
            for (int64_t ti = 0; ti < length; ++ti) {
                const int64_t u_off = (bi * length + ti) * dim + di;
                const int64_t bc_off = (bi * length + ti) * state;
                float acc = 0.0f;
                for (int64_t ni = 0; ni < state; ++ni) {
                    const int64_t state_off = (bi * dim + di) * state + ni;
                    const float decay = std::exp(delta[u_off] * a[di * state + ni]);
                    const float update = delta[u_off] * b[bc_off + ni] * u[u_off];
                    state_value[state_off] = decay * state_value[state_off] + update;
                    acc += state_value[state_off] * c[bc_off + ni];
                }
                output[u_off] = acc + u[u_off] * d[di];
            }
        }
    }
    return output;
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(lhs[i] - rhs[i]));
    }
    return max_abs_diff;
}

void CopyHostToDevice(void* dst, const std::vector<float>& src)
{
    CHECK_ACL(aclrtMemcpy(dst, src.size() * sizeof(float), src.data(), src.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));
}

}  // namespace

int main(int argc, char** argv)
{
    const int device_id = static_cast<int>(Arg(argc, argv, 1, 0));
    const int64_t batch = Arg(argc, argv, 2, 1);
    const int64_t length = Arg(argc, argv, 3, 1024);
    const int64_t dim = Arg(argc, argv, 4, 1536);
    const int64_t state = Arg(argc, argv, 5, 16);
    const int64_t repeat = Arg(argc, argv, 6, 10);
    const int64_t warmup = Arg(argc, argv, 7, 3);
    if (std::min({batch, length, dim, state, repeat}) <= 0 || warmup < 0) {
        std::cerr << "Usage: benchmark_selective_scan_1d <device_id> <batch> <length> <dim> <state> <repeat> <warmup>"
                  << std::endl;
        return 2;
    }

    std::mt19937 rng(2026);
    std::vector<float> u = RandomVector(static_cast<size_t>(batch * length * dim), -0.25f, 0.25f, rng);
    std::vector<float> delta = RandomVector(static_cast<size_t>(batch * length * dim), 0.01f, 0.08f, rng);
    std::vector<float> a = RandomVector(static_cast<size_t>(dim * state), -0.20f, -0.01f, rng);
    std::vector<float> b = RandomVector(static_cast<size_t>(batch * length * state), -0.20f, 0.20f, rng);
    std::vector<float> c = RandomVector(static_cast<size_t>(batch * length * state), -0.20f, 0.20f, rng);
    std::vector<float> d = RandomVector(static_cast<size_t>(dim), -0.05f, 0.05f, rng);

    const auto cpu_start = std::chrono::steady_clock::now();
    std::vector<float> expected = CpuReference(u, delta, a, b, c, d, batch, length, dim, state);
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void* d_u = MallocDevice(u.size() * sizeof(float));
    void* d_delta = MallocDevice(delta.size() * sizeof(float));
    void* d_a = MallocDevice(a.size() * sizeof(float));
    void* d_b = MallocDevice(b.size() * sizeof(float));
    void* d_c = MallocDevice(c.size() * sizeof(float));
    void* d_d = MallocDevice(d.size() * sizeof(float));
    void* d_output = MallocDevice(expected.size() * sizeof(float));
    const uint64_t workspace_size = aclnnSelectiveScan1DGetWorkspaceSize(batch, length, dim, state);
    void* workspace = MallocDevice(workspace_size);

    CopyHostToDevice(d_u, u);
    CopyHostToDevice(d_delta, delta);
    CopyHostToDevice(d_a, a);
    CopyHostToDevice(d_b, b);
    CopyHostToDevice(d_c, c);
    CopyHostToDevice(d_d, d);

    for (int64_t i = 0; i < warmup; ++i) {
        CHECK_ACL(static_cast<aclError>(aclnnSelectiveScan1D(
            d_u, d_delta, d_a, d_b, d_c, d_d, d_output, batch, length, dim, state, workspace, workspace_size, stream)));
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    CHECK_ACL(aclrtCreateEvent(&start));
    CHECK_ACL(aclrtCreateEvent(&end));
    CHECK_ACL(aclrtRecordEvent(start, stream));
    for (int64_t i = 0; i < repeat; ++i) {
        CHECK_ACL(static_cast<aclError>(aclnnSelectiveScan1D(
            d_u, d_delta, d_a, d_b, d_c, d_d, d_output, batch, length, dim, state, workspace, workspace_size, stream)));
    }
    CHECK_ACL(aclrtRecordEvent(end, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    float npu_total_ms = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&npu_total_ms, start, end));

    std::vector<float> actual(expected.size(), 0.0f);
    CHECK_ACL(aclrtMemcpy(actual.data(), actual.size() * sizeof(float), d_output, actual.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    const float max_abs_diff = MaxAbsDiff(actual, expected);
    const double npu_hot_ms = static_cast<double>(npu_total_ms) / static_cast<double>(repeat);

    std::cout << "name,shape,cpu_ref_ms,npu_hot_ms,max_abs_diff,npu_hot_vs_cpu" << std::endl;
    std::cout << "SelectiveScan1D," << batch << "x" << length << "x" << dim << "xN" << state << "," << cpu_ms
              << "," << npu_hot_ms << "," << max_abs_diff << "," << (cpu_ms / npu_hot_ms) << std::endl;

    aclrtDestroyEvent(end);
    aclrtDestroyEvent(start);
    aclrtFree(workspace);
    aclrtFree(d_output);
    aclrtFree(d_d);
    aclrtFree(d_c);
    aclrtFree(d_b);
    aclrtFree(d_a);
    aclrtFree(d_delta);
    aclrtFree(d_u);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return max_abs_diff <= 2.0e-4f ? 0 : 2;
}
