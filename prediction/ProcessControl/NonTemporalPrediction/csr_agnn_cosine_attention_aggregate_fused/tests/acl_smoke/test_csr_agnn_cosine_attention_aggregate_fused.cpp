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

#include "csr_agnn_cosine_attention_aggregate_fused_host.h"

namespace
{
void Check(aclError error, const char* name)
{
    if (error != ACL_SUCCESS)
    {
        std::fprintf(stderr, "%s failed: %d\n", name, static_cast<int>(error));
        std::exit(1);
    }
}
void CopyIn(void** device, const void* host, size_t bytes)
{
    Check(aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
    Check(aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy");
}
int VerifyOutput(const std::vector<float>& output)
{
    const std::vector<float> expected{0.7310586F, 0.5378828F, 1.0F, 0.0F};
    for (size_t index = 0; index < output.size(); ++index)
    {
        if (std::fabs(output[index] - expected[index]) > 1.0e-4F)
        {
            std::fprintf(stderr, "mismatch %zu: %.6f != %.6f\n", index, output[index], expected[index]);
            return 1;
        }
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    int device = argc > 1 ? std::atoi(argv[1]) : 0;
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "set device");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "stream");
    const std::vector<int32_t> row{0, 2, 3};
    const std::vector<int32_t> source{0, 1, 0};
    const std::vector<float> features{1.0F, 0.0F, 0.0F, 2.0F};
    const std::vector<float> normalized{1.0F, 0.0F, 0.0F, 1.0F};
    std::vector<float> output(4, 0.0F);
    void *dRow = nullptr, *dSource = nullptr, *dFeatures = nullptr;
    void *dNormalized = nullptr, *dOutput = nullptr, *workspace = nullptr;
    CopyIn(&dRow, row.data(), row.size() * sizeof(int32_t));
    CopyIn(&dSource, source.data(), source.size() * sizeof(int32_t));
    CopyIn(&dFeatures, features.data(), features.size() * sizeof(float));
    CopyIn(&dNormalized, normalized.data(), normalized.size() * sizeof(float));
    Check(aclrtMalloc(&dOutput, output.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "output");
    uint64_t bytes = aclnnCsrAgnnCosineAttentionAggregateFusedGetWorkspaceSize(2, 3, 2, 2);
    Check(aclrtMalloc(&workspace, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "workspace");
    if (aclnnCsrAgnnCosineAttentionAggregateFused(dRow, dSource, dFeatures, dNormalized, dOutput, 2, 3, 2, 2049, 1.0F,
                                                  workspace, bytes, stream) != ACL_ERROR_INVALID_PARAM)
    {
        return 1;
    }
    if (aclnnCsrAgnnCosineAttentionAggregateFused(dRow, dSource, dFeatures, dNormalized, dOutput, 2, 3, 2, 2, 1.0F,
                                                  workspace, bytes, stream) != ACL_SUCCESS)
    {
        return 1;
    }
    Check(aclrtSynchronizeStream(stream), "sync");
    Check(aclrtMemcpy(output.data(), output.size() * sizeof(float), dOutput, output.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "copy out");
    const int result = VerifyOutput(output);
    aclrtFree(workspace);
    aclrtFree(dOutput);
    aclrtFree(dNormalized);
    aclrtFree(dFeatures);
    aclrtFree(dSource);
    aclrtFree(dRow);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return result;
}
