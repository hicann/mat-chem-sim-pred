/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <acl/acl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dimenet_triplet_angle_fused_host.h"

namespace
{
void Check(aclError value, const char* name)
{
    if (value != ACL_SUCCESS)
    {
        std::fprintf(stderr, "%s failed\n", name);
        std::exit(1);
    }
}
void CopyIn(void** device, const void* host, size_t bytes)
{
    Check(aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
    Check(aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy in");
}
}  // namespace

int main(int argc, char** argv)
{
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    Check(aclInit(nullptr), "init");
    Check(aclrtSetDevice(device), "device");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "stream");
    const std::vector<float> position{0., 0., 0., 1., 0., 0., 1., 1., 0.};
    const std::vector<int32_t> idxI{0}, idxJ{1}, idxK{2};
    float output = -1.0F;
    void *dPosition = nullptr, *dI = nullptr, *dJ = nullptr, *dK = nullptr, *dOutput = nullptr;
    CopyIn(&dPosition, position.data(), position.size() * sizeof(float));
    CopyIn(&dI, idxI.data(), sizeof(int32_t));
    CopyIn(&dJ, idxJ.data(), sizeof(int32_t));
    CopyIn(&dK, idxK.data(), sizeof(int32_t));
    Check(aclrtMalloc(&dOutput, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "output");
    const uint64_t workspace = aclnnDimeNetTripletAngleFusedGetWorkspaceSize(3, 1);
    if (aclnnDimeNetTripletAngleFused(dPosition, dI, dJ, dK, dOutput, 3, 1, nullptr, workspace, stream) != ACL_SUCCESS)
        return 2;
    Check(aclrtSynchronizeStream(stream), "sync");
    Check(aclrtMemcpy(&output, sizeof(float), dOutput, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), "copy out");
    const bool passed = std::fabs(output - 1.5707963F) < 2.0e-4F;
    aclrtFree(dOutput);
    aclrtFree(dK);
    aclrtFree(dJ);
    aclrtFree(dI);
    aclrtFree(dPosition);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return passed ? 0 : 10;
}
