/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cstdint>
#include <vector>

#include "../../graph_message_example_common.h"
#include "csr_film_modulated_mean_fused_host.h"

int main(int argc, char** argv)
{
    int device = 0;
    if (!GraphMessageExample::ParseDevice(argc, argv, &device))
    {
        return 1;
    }
    GraphMessageExample::Runtime runtime(device);
    GraphMessageExample::Buffers buffers;
    const std::vector<int32_t> rowPtr{0, 2, 3, 3};
    const std::vector<int32_t> source{0, 1, 0};
    const std::vector<float> projected{1, -2, 3, 4, 5, 6};
    const std::vector<float> beta{0.5F, 1, -1, 0.5F, 2, 2};
    const std::vector<float> gamma{2, -1, 0.5F, 2, 1, 1};
    const std::vector<float> expected{4.5F, 1.5F, 0, 0, 0, 0};
    std::vector<float> output(expected.size(), 0.0F);
    void* dRow = buffers.Copy(rowPtr);
    void* dSource = buffers.Copy(source);
    void* dProjected = buffers.Copy(projected);
    void* dBeta = buffers.Copy(beta);
    void* dGamma = buffers.Copy(gamma);
    void* dOutput = buffers.Allocate(output.size() * sizeof(float));
    const uint64_t workspaceSize = aclnnCsrFilmModulatedMeanFusedGetWorkspaceSize(3, 3, 2, 2);
    void* workspace = buffers.Allocate(workspaceSize);
    bool passed = aclnnCsrFilmModulatedMeanFused(dRow, dSource, dProjected, dBeta, dGamma, dOutput, 3, 3, 513, 2, 1,
                                                 workspace, workspaceSize, runtime.Stream()) == ACL_ERROR_INVALID_PARAM;
    GraphMessageExample::Check(aclnnCsrFilmModulatedMeanFused(dRow, dSource, dProjected, dBeta, dGamma, dOutput, 3, 3,
                                                              2, 2, 1, workspace, workspaceSize, runtime.Stream()),
                               "launch");
    GraphMessageExample::CopyOut(&output, dOutput, runtime.Stream());
    passed &= GraphMessageExample::Matches(output, expected);
    return passed ? 0 : 1;
}
