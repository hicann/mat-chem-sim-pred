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

#include "csr_point_transformer_attention_aggregate_fused_host.h"

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
    Check(aclrtCreateStream(&stream), "stream");
    const std::vector<int32_t> row{0, 2, 3}, source{0, 1, 0};
    const std::vector<float> alphaSource{0.0F, 1.0F};
    const std::vector<float> alphaTarget{0.0F, 1.0F};
    const std::vector<float> value{2.0F, 4.0F}, delta{0.0F, 0.0F, 0.0F};
    const std::vector<float> expected{2.537883F, 2.0F};
    std::vector<float> output(2, 0.0F);
    void *dRow = nullptr, *dSource = nullptr, *dAlphaSource = nullptr;
    void *dAlphaTarget = nullptr, *dValue = nullptr, *dDelta = nullptr;
    void *dOutput = nullptr, *workspace = nullptr;
    CopyIn(&dRow, row.data(), row.size() * sizeof(int32_t));
    CopyIn(&dSource, source.data(), source.size() * sizeof(int32_t));
    CopyIn(&dAlphaSource, alphaSource.data(), alphaSource.size() * sizeof(float));
    CopyIn(&dAlphaTarget, alphaTarget.data(), alphaTarget.size() * sizeof(float));
    CopyIn(&dValue, value.data(), value.size() * sizeof(float));
    CopyIn(&dDelta, delta.data(), delta.size() * sizeof(float));
    Check(aclrtMalloc(&dOutput, output.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "output");
    const uint64_t bytes = aclnnCsrPointTransformerAttentionAggregateFusedGetWorkspaceSize(2, 3, 1, 2);
    Check(aclrtMalloc(&workspace, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "workspace");
    Check(aclnnCsrPointTransformerAttentionAggregateFused(dRow, dSource, dAlphaSource, dAlphaTarget, dValue, dDelta,
                                                          dOutput, 2, 3, 1, 2, 0, workspace, bytes, stream),
          "launch");
    Check(aclrtSynchronizeStream(stream), "synchronize");
    Check(aclrtMemcpy(output.data(), output.size() * sizeof(float), dOutput, output.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "copy out");
    bool passed = true;
    for (size_t i = 0; i < output.size(); ++i) passed &= std::fabs(output[i] - expected[i]) <= 1.0e-5F;
    aclrtFree(workspace);
    aclrtFree(dOutput);
    aclrtFree(dDelta);
    aclrtFree(dValue);
    aclrtFree(dAlphaTarget);
    aclrtFree(dAlphaSource);
    aclrtFree(dSource);
    aclrtFree(dRow);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return passed ? 0 : 1;
}
