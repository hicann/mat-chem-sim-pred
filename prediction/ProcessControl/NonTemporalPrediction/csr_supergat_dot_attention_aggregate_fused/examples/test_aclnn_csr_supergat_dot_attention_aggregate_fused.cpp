/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <acl/acl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "csr_supergat_dot_attention_aggregate_fused_host.h"

namespace
{
void Check(aclError result, const char* operation)
{
    if (result != ACL_SUCCESS)
    {
        std::fprintf(stderr, "%s failed: %d\n", operation, result);
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
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "set device");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "create stream");
    const std::vector<int32_t> row{0, 2, 3}, source{0, 1, 0};
    const std::vector<float> projected{1.0F, 3.0F};
    const std::vector<float> expected{2.761594F, 1.0F};
    std::vector<float> output(2, 0.0F);
    void *dRow = nullptr, *dSource = nullptr, *dProjected = nullptr;
    void *dOutput = nullptr, *workspace = nullptr;
    CopyIn(&dRow, row.data(), row.size() * sizeof(int32_t));
    CopyIn(&dSource, source.data(), source.size() * sizeof(int32_t));
    CopyIn(&dProjected, projected.data(), projected.size() * sizeof(float));
    Check(aclrtMalloc(&dOutput, output.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "output");
    const uint64_t bytes = aclnnCsrSuperGatDotAttentionAggregateFusedGetWorkspaceSize(2, 3, 1, 1, 2);
    Check(aclrtMalloc(&workspace, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "workspace");
    Check(aclnnCsrSuperGatDotAttentionAggregateFused(dRow, dSource, dProjected, dOutput, 2, 3, 1, 1, 2, 0, 0.2F,
                                                     workspace, bytes, stream),
          "launch");
    Check(aclrtSynchronizeStream(stream), "synchronize");
    Check(aclrtMemcpy(output.data(), output.size() * sizeof(float), dOutput, output.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "copy out");
    bool passed = true;
    for (size_t i = 0; i < output.size(); ++i)
    {
        passed &= std::fabs(output[i] - expected[i]) <= 1.0e-5F;
    }
    aclrtFree(workspace);
    aclrtFree(dOutput);
    aclrtFree(dProjected);
    aclrtFree(dSource);
    aclrtFree(dRow);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return passed ? 0 : 1;
}
