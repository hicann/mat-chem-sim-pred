/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <acl/acl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ppf_point_pair_features_fused_host.h"

namespace
{
void RequireSuccess(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS)
    {
        std::fprintf(stderr, "ACL operation failed: %s\n", operation);
        std::exit(1);
    }
}

void* CopyToDevice(const void* host, size_t bytes)
{
    void* device = nullptr;
    RequireSuccess(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate input");
    RequireSuccess(aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload input");
    return device;
}
}  // namespace

int main(int argc, char** argv)
{
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    RequireSuccess(aclInit(nullptr), "initialize runtime");
    RequireSuccess(aclrtSetDevice(device), "select device");
    aclrtStream stream = nullptr;
    RequireSuccess(aclrtCreateStream(&stream), "create stream");
    const std::vector<float> position{0., 0., 0., 1., 0., 0.};
    const std::vector<float> normal{0., 1., 0., 0., 1., 0.};
    const std::vector<int32_t> source{1}, target{0};
    std::vector<float> output(4, -1.0F);
    void* dPosition = CopyToDevice(position.data(), position.size() * sizeof(float));
    void* dNormal = CopyToDevice(normal.data(), normal.size() * sizeof(float));
    void* dSource = CopyToDevice(source.data(), sizeof(int32_t));
    void* dTarget = CopyToDevice(target.data(), sizeof(int32_t));
    void* dOutput = nullptr;
    RequireSuccess(aclrtMalloc(&dOutput, output.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "allocate output");
    const uint64_t workspace = aclnnPpfPointPairFeaturesFusedGetWorkspaceSize(2, 1);
    if (aclnnPpfPointPairFeaturesFused(dPosition, dNormal, dSource, dTarget, dOutput, 2, 1, nullptr, workspace,
                                       stream) != ACL_SUCCESS)
    {
        return 2;
    }
    RequireSuccess(aclrtSynchronizeStream(stream), "synchronize stream");
    RequireSuccess(aclrtMemcpy(output.data(), output.size() * sizeof(float), dOutput, output.size() * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST),
                   "download output");
    const bool passed = std::fabs(output[0] - 1.0F) < 1.0e-5F && std::fabs(output[1] - 1.5707963F) < 2.0e-4F &&
                        std::fabs(output[2] - 1.5707963F) < 2.0e-4F && std::fabs(output[3]) < 2.0e-4F;
    aclrtFree(dOutput);
    aclrtFree(dTarget);
    aclrtFree(dSource);
    aclrtFree(dNormal);
    aclrtFree(dPosition);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return passed ? 0 : 10;
}
