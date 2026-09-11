/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include "csr_appnp_propagate_fused_host.h"
#include "example_common.h"

bool RunAppnpCase(const std::array<void*, 5>& devices, void* workspace, uint64_t workspaceSize, aclrtStream stream,
                  std::array<float, 4>& output)
{
    const std::vector<int32_t> rowPtr{0, 2, 3};
    const std::vector<int32_t> columnIndex{0, 1, 0};
    const std::vector<float> edgeWeight{0.5F, 0.5F, 1.0F};
    const std::vector<float> initial{2.0F, -2.0F, 4.0F, 6.0F};
    bool passed = AllChecks({
        CopyToDevice(devices[0], rowPtr.size() * sizeof(int32_t), rowPtr.data(), "copy row_ptr"),
        CopyToDevice(devices[1], columnIndex.size() * sizeof(int32_t), columnIndex.data(), "copy column_index"),
        CopyToDevice(devices[2], edgeWeight.size() * sizeof(float), edgeWeight.data(), "copy edge_weight"),
        CopyToDevice(devices[3], initial.size() * sizeof(float), initial.data(), "copy initial"),
    });
    passed &= aclnnCsrAppnpPropagateFused(devices[0], devices[1], devices[2], devices[3], devices[4], 2, 3, 2, 2, 0.25F,
                                          workspace, workspaceSize - 1U, stream) == ACL_ERROR_INVALID_PARAM;
    passed &= aclnnCsrAppnpPropagateFused(devices[0], devices[1], devices[2], devices[3], devices[4], 2, 3, 2, 2,
                                          std::numeric_limits<float>::infinity(), workspace, workspaceSize,
                                          stream) == ACL_ERROR_INVALID_PARAM;
    passed &= aclnnCsrAppnpPropagateFused(devices[0], devices[1], devices[2], devices[3], devices[3], 2, 3, 2, 2, 0.25F,
                                          workspace, workspaceSize, stream) == ACL_ERROR_INVALID_PARAM;
    passed &= CheckAcl(aclnnCsrAppnpPropagateFused(devices[0], devices[1], devices[2], devices[3], devices[4], 2, 3, 2,
                                                   2, 0.25F, workspace, workspaceSize, stream),
                       "operator launch");
    const std::array<float, 4> expected{2.46875F, -0.125F, 3.0625F, 2.25F};
    passed &= CopyAndCheckOutput(output, devices[4], stream, expected);
    return passed;
}

int main(int argc, char** argv)
{
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    if (!CheckAcl(aclInit(nullptr), "aclInit") || !CheckAcl(aclrtSetDevice(device), "aclrtSetDevice")) return 1;
    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) return 1;
    std::array<float, 4> output{};
    const std::array<void*, 5> devices{AllocateAcl(3 * sizeof(int32_t)), AllocateAcl(3 * sizeof(int32_t)),
                                       AllocateAcl(3 * sizeof(float)), AllocateAcl(4 * sizeof(float)),
                                       AllocateAcl(output.size() * sizeof(float))};
    const uint64_t workspaceSize = aclnnCsrAppnpPropagateFusedGetWorkspaceSize(2, 3, 2, 2);
    void* workspace = AllocateAcl(workspaceSize);
    if (std::any_of(devices.begin(), devices.end(), [](void* pointer) { return pointer == nullptr; }) ||
        workspace == nullptr)
        return 1;
    const bool passed = RunAppnpCase(devices, workspace, workspaceSize, stream, output);
    std::cout << "outputs=" << output[0] << ',' << output[1] << ',' << output[2] << ',' << output[3]
              << (passed ? " PASSED" : " FAILED") << std::endl;
    aclrtFree(workspace);
    for (void* pointer : devices) aclrtFree(pointer);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return passed ? 0 : 1;
}
