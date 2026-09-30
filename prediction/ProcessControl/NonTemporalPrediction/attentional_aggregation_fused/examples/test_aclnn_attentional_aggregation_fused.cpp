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

#include "attentional_aggregation_fused_host.h"

namespace
{
bool Check(aclError status, const char* stage)
{
    if (status == ACL_SUCCESS) return true;
    std::cerr << stage << " failed with ACL error " << status << std::endl;
    return false;
}

void* Allocate(size_t bytes)
{
    void* pointer = nullptr;
    return Check(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc") ? pointer : nullptr;
}

bool ParseDevice(int argc, char** argv, int* device)
{
    *device = 0;
    if (argc <= 1) return true;
    char* end = nullptr;
    const long parsed = std::strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || parsed < 0 || parsed > 1024)
    {
        std::fprintf(stderr, "invalid device id: %s\n", argv[1]);
        return false;
    }
    *device = static_cast<int>(parsed);
    return true;
}

void Cleanup(int device, aclrtStream stream, void* workspace, void* output, void* gates, void* features, void* graphPtr)
{
    aclrtFree(workspace);
    aclrtFree(output);
    aclrtFree(gates);
    aclrtFree(features);
    aclrtFree(graphPtr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
}

bool CopyInputs(void* graphPtr, size_t graphBytes, const void* graph, void* features, size_t featureBytes,
                const void* featureData, void* gates, size_t gateBytes, const void* gateData)
{
    return Check(aclrtMemcpy(graphPtr, graphBytes, graph, graphBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy graph_ptr") &&
           Check(aclrtMemcpy(features, featureBytes, featureData, featureBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                 "copy features") &&
           Check(aclrtMemcpy(gates, gateBytes, gateData, gateBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy gates");
}
}  // namespace

int main(int argc, char** argv)
{
    int device = 0;
    if (!ParseDevice(argc, argv, &device)) return 1;
    if (!Check(aclInit(nullptr), "aclInit") || !Check(aclrtSetDevice(device), "aclrtSetDevice")) return 1;
    aclrtStream stream = nullptr;
    if (!Check(aclrtCreateStream(&stream), "aclrtCreateStream")) return 1;
    const std::vector<int32_t> graphPtr{0, 2, 3};
    const std::vector<float> features{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::vector<float> gates{0.0F, 0.0F, -3.0F};
    std::array<float, 4> output{};
    void* deviceGraphPtr = Allocate(graphPtr.size() * sizeof(int32_t));
    void* deviceFeatures = Allocate(features.size() * sizeof(float));
    void* deviceGates = Allocate(gates.size() * sizeof(float));
    void* deviceOutput = Allocate(output.size() * sizeof(float));
    const uint64_t workspaceSize = aclnnAttentionalAggregationFusedGetWorkspaceSize(3, 2, 2);
    void* workspace = Allocate(workspaceSize);
    if (!deviceGraphPtr || !deviceFeatures || !deviceGates || !deviceOutput || !workspace)
    {
        Cleanup(device, stream, workspace, deviceOutput, deviceGates, deviceFeatures, deviceGraphPtr);
        return 1;
    }
    bool passed = true;
    passed &= CopyInputs(deviceGraphPtr, graphPtr.size() * sizeof(int32_t), graphPtr.data(), deviceFeatures,
                         features.size() * sizeof(float), features.data(), deviceGates, gates.size() * sizeof(float),
                         gates.data());
    passed &= aclnnAttentionalAggregationFused(deviceGraphPtr, deviceFeatures, deviceGates, deviceOutput, 3, 2, 2,
                                               workspace, workspaceSize - 1U, stream) == ACL_ERROR_INVALID_PARAM;
    passed &= aclnnAttentionalAggregationFused(nullptr, deviceFeatures, deviceGates, deviceOutput, 3, 2, 2, workspace,
                                               workspaceSize, stream) == ACL_ERROR_INVALID_PARAM;
    passed &= Check(aclnnAttentionalAggregationFused(deviceGraphPtr, deviceFeatures, deviceGates, deviceOutput, 3, 2, 2,
                                                     workspace, workspaceSize, stream),
                    "operator launch");
    passed &= Check(aclrtSynchronizeStream(stream), "synchronize");
    passed &= Check(aclrtMemcpy(output.data(), output.size() * sizeof(float), deviceOutput,
                                output.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
                    "copy output");
    const std::array<float, 4> expected{2.0F, 3.0F, 5.0F, 6.0F};
    for (size_t index = 0; index < output.size(); ++index)
    {
        passed &= std::fabs(output[index] - expected[index]) < 1.0e-5F;
    }
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    Cleanup(device, stream, workspace, deviceOutput, deviceGates, deviceFeatures, deviceGraphPtr);
    return passed ? 0 : 1;
}
