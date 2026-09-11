/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "../../csr_appnp_propagate_fused/examples/example_common.h"
#include "csr_sage_mean_root_relu_fused_host.h"

bool RunSageCase(const std::array<void*, 5>& devices, void* workspace, uint64_t workspaceSize, aclrtStream stream,
                 std::array<float, 4>& output)
{
    const std::vector<int32_t> rowPtr{0, 2, 3};
    const std::vector<int32_t> columnIndex{0, 1, 0};
    const std::vector<float> neighbor{4.0F, -1.0F, 0.0F, 3.0F};
    const std::vector<float> root{1.0F, 2.0F, -2.0F, -1.0F};
    bool passed = AllChecks(
        {CopyToDevice(devices[0], rowPtr.size() * sizeof(int32_t), rowPtr.data(), "copy row_ptr"),
         CopyToDevice(devices[1], columnIndex.size() * sizeof(int32_t), columnIndex.data(), "copy column_index"),
         CopyToDevice(devices[2], neighbor.size() * sizeof(float), neighbor.data(), "copy neighbor"),
         CopyToDevice(devices[3], root.size() * sizeof(float), root.data(), "copy root")});
    passed &= aclnnCsrSageMeanRootReluFused(devices[0], devices[1], devices[2], devices[3], devices[4], 2, 3, 2,
                                            workspace, workspaceSize - 1U, stream) == ACL_ERROR_INVALID_PARAM;
    passed &= aclnnCsrSageMeanRootReluFused(devices[0], devices[1], devices[2], devices[3], devices[4], 2, 3, 1025,
                                            workspace, workspaceSize, stream) == ACL_ERROR_INVALID_PARAM;
    passed &= CheckAcl(aclnnCsrSageMeanRootReluFused(devices[0], devices[1], devices[2], devices[3], devices[4], 2, 3,
                                                     2, workspace, workspaceSize, stream),
                       "operator launch");
    passed &= CheckAcl(aclrtSynchronizeStream(stream), "sage operator completion");
    const char* resultLabel = "sage_mean_root_relu";
    const std::array<float, 4> expected{3.0F, 3.0F, 2.0F, 0.0F};
    passed = CopyAndCheckOutput(output, devices[4], stream, expected) && passed;
    if (!passed) std::cerr << resultLabel << " output validation failed" << std::endl;
    return passed;
}

int main(int argc, char** argv)
{
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    aclrtStream stream = nullptr;
    if (!InitializeAclDevice(device, stream)) return 1;
    std::array<float, 4> output{};
    const std::array<void*, 5> devices{AllocateAcl(3 * sizeof(int32_t)), AllocateAcl(3 * sizeof(int32_t)),
                                       AllocateAcl(4 * sizeof(float)), AllocateAcl(4 * sizeof(float)),
                                       AllocateAcl(output.size() * sizeof(float))};
    const uint64_t workspaceSize = aclnnCsrSageMeanRootReluFusedGetWorkspaceSize(2, 3, 2);
    void* workspace = AllocateAcl(workspaceSize);
    if (std::any_of(devices.begin(), devices.end(), [](void* pointer) { return pointer == nullptr; }) ||
        workspace == nullptr)
        return 1;
    const bool passed = RunSageCase(devices, workspace, workspaceSize, stream, output);
    std::cout << "outputs=" << output[0] << ',' << output[1] << ',' << output[2] << ',' << output[3]
              << (passed ? " PASSED" : " FAILED") << std::endl;
    aclrtFree(workspace);
    for (void* pointer : devices) aclrtFree(pointer);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return passed ? 0 : 1;
}
