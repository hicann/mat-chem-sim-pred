/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "sort_aggregation_topk_fused_host.h"

namespace
{
bool CheckSort(aclError result, const char* stage)
{
    if (result == ACL_SUCCESS) return true;
    std::cerr << stage << " failed with ACL status " << result << std::endl;
    return false;
}

void* Allocate(size_t bytes)
{
    void* pointer = nullptr;
    return CheckSort(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc") ? pointer : nullptr;
}

bool ParseSortDevice(int argc, char** argv, int* device)
{
    *device = 0;
    constexpr long kSortMaxDevice = 1024;
    if (argc <= 1) return true;
    char* end = nullptr;
    const long parsed = std::strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || parsed < 0 || parsed > kSortMaxDevice)
    {
        std::fprintf(stderr, "invalid device id: %s\n", argv[1]);
        return false;
    }
    *device = static_cast<int>(parsed);
    return true;
}

void CleanupSort(int device, aclrtStream stream, void* workspace, void* output, void* features, void* graphPtr)
{
    aclrtFree(workspace);
    aclrtFree(output);
    aclrtFree(features);
    aclrtFree(graphPtr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
}

bool CopySortInputs(void* graphPtr, size_t graphBytes, const void* graph, void* features, size_t featureBytes,
                    const void* featureData)
{
    return CheckSort(aclrtMemcpy(graphPtr, graphBytes, graph, graphBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                     "sort copy graph_ptr") &&
           CheckSort(aclrtMemcpy(features, featureBytes, featureData, featureBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                     "copy features");
}
}  // namespace

int main(int argc, char** argv)
{
    int device = 0;
    if (!ParseSortDevice(argc, argv, &device)) return 1;
    if (!CheckSort(aclInit(nullptr), "aclInit") || !CheckSort(aclrtSetDevice(device), "aclrtSetDevice")) return 1;
    aclrtStream stream = nullptr;
    if (!CheckSort(aclrtCreateStream(&stream), "aclrtCreateStream")) return 1;

    const std::vector<int32_t> graphPtr{0, 3, 5};
    const std::vector<float> features{1.0F, 0.2F, 2.0F, 0.8F, 3.0F, -0.1F, 4.0F, 0.4F, 5.0F, 0.3F};
    std::array<float, 12> output{};
    void* deviceGraphPtr = Allocate(graphPtr.size() * sizeof(int32_t));
    void* deviceFeatures = Allocate(features.size() * sizeof(float));
    void* deviceOutput = Allocate(output.size() * sizeof(float));
    const uint64_t workspaceSize = aclnnSortAggregationTopkFusedGetWorkspaceSize(5, 2, 2, 3);
    void* workspace = Allocate(workspaceSize);
    if (!deviceGraphPtr || !deviceFeatures || !deviceOutput || !workspace)
    {
        CleanupSort(device, stream, workspace, deviceOutput, deviceFeatures, deviceGraphPtr);
        return 1;
    }

    bool passed = true;
    passed &= CopySortInputs(deviceGraphPtr, graphPtr.size() * sizeof(int32_t), graphPtr.data(), deviceFeatures,
                             features.size() * sizeof(float), features.data());
    passed &= aclnnSortAggregationTopkFused(deviceGraphPtr, deviceFeatures, deviceOutput, 5, 2, 2, 3, workspace,
                                            workspaceSize - 1U, stream) == ACL_ERROR_INVALID_PARAM;
    const int64_t invalidTopK = 1025;
    passed &= aclnnSortAggregationTopkFused(deviceGraphPtr, deviceFeatures, deviceOutput, 5, 2, invalidTopK, 3,
                                            workspace, workspaceSize, stream) == ACL_ERROR_INVALID_PARAM;
    passed &= aclnnSortAggregationTopkFused(deviceGraphPtr, deviceFeatures, deviceOutput, 5, 2, 2, 129, workspace,
                                            workspaceSize, stream) == ACL_ERROR_INVALID_PARAM;
    passed &= aclnnSortAggregationTopkFused(nullptr, deviceFeatures, deviceOutput, 5, 2, 2, 3, workspace, workspaceSize,
                                            stream) == ACL_ERROR_INVALID_PARAM;
    passed &= CheckSort(aclnnSortAggregationTopkFused(deviceGraphPtr, deviceFeatures, deviceOutput, 5, 2, 2, 3,
                                                      workspace, workspaceSize, stream),
                        "operator launch");
    passed &= CheckSort(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    passed &= CheckSort(aclrtMemcpy(output.data(), output.size() * sizeof(float), deviceOutput,
                                    output.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
                        "copy output");

    const std::array<float, 12> expected{2.0F, 0.8F, 1.0F, 0.2F, 3.0F, -0.1F, 4.0F, 0.4F, 5.0F, 0.3F, 0.0F, 0.0F};
    for (size_t index = 0; index < output.size(); ++index)
    {
        passed &= std::fabs(output[index] - expected[index]) < 1.0e-6F;
    }
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    CleanupSort(device, stream, workspace, deviceOutput, deviceFeatures, deviceGraphPtr);
    return passed ? 0 : 1;
}
