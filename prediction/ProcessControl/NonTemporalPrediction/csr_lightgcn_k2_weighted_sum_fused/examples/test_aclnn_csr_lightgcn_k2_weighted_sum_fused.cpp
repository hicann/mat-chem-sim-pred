/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cmath>
#include <cstdint>
#include <vector>

#include "../../graph_message_example_common.h"
#include "csr_lightgcn_k2_weighted_sum_fused_host.h"

int main(int argc, char** argv)
{
    int device = 0;
    if (!GraphMessageExample::ParseDevice(argc, argv, &device))
    {
        return 1;
    }
    GraphMessageExample::Runtime runtime(device);
    GraphMessageExample::Buffers buffers;
    const std::vector<int32_t> row{0, 2, 3};
    const std::vector<int32_t> source{1, 0, 1};
    const std::vector<float> norm{0.5F, 0.25F, 0.75F};
    const std::vector<float> features{1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> output(4, 0.0F);
    void* dRow = buffers.Copy(row);
    void* dSource = buffers.Copy(source);
    void* dNorm = buffers.Copy(norm);
    void* dFeatures = buffers.Copy(features);
    void* dOutput = buffers.Allocate(output.size() * sizeof(float));
    const uint64_t bytes = aclnnCsrLightgcnK2WeightedSumFusedGetWorkspaceSize(2, 3, 2);
    void* workspace = buffers.Allocate(bytes);
    bool passed =
        aclnnCsrLightgcnK2WeightedSumFused(dRow, dSource, dNorm, dFeatures, dOutput, 2, 3, 2, 1.0F, std::nanf(""), 1.0F,
                                           workspace, bytes, runtime.Stream()) == ACL_ERROR_INVALID_PARAM;
    passed &= aclnnCsrLightgcnK2WeightedSumFused(dRow, dSource, dNorm, dFeatures, dOutput, 2, 3, 2, 0.2F, 0.3F, 0.5F,
                                                 workspace, bytes, runtime.Stream()) == ACL_SUCCESS;
    GraphMessageExample::CopyOut(&output, dOutput, runtime.Stream());
    for (float value : output)
    {
        passed &= std::isfinite(value);
    }
    return passed ? 0 : 1;
}
