/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cstdio>
#include <vector>

#include "../../attention_example_common.h"
#include "csr_gat_attention_aggregate_fused_host.h"

int main(int argc, char** argv)
{
    int device = 0;
    if (!AttentionExample::ParseDevice(argc, argv, &device))
    {
        return 1;
    }
    AttentionExample::Runtime runtime(device);
    AttentionExample::Buffers buffers;
    const std::vector<int32_t> rowPtr{0, 2, 3};
    const std::vector<int32_t> source{0, 1, 0};
    const std::vector<float> projected{1.0F, 3.0F};
    const std::vector<float> attentionSource{1.0F};
    const std::vector<float> attentionTarget{0.0F};
    const std::vector<float> expected{2.761594F, 1.0F};
    std::vector<float> output(expected.size(), 0.0F);
    void* dRow = buffers.Copy(rowPtr);
    void* dSource = buffers.Copy(source);
    void* dProjected = buffers.Copy(projected);
    void* dAttentionSource = buffers.Copy(attentionSource);
    void* dAttentionTarget = buffers.Copy(attentionTarget);
    void* dOutput = buffers.Allocate(output.size() * sizeof(float));
    const uint64_t workspaceSize = aclnnCsrGatAttentionAggregateFusedGetWorkspaceSize(2, 3, 1, 1, 2);
    void* workspace = buffers.Allocate(workspaceSize);
    bool passed = aclnnCsrGatAttentionAggregateFused(dRow, dSource, dProjected, dAttentionSource, dAttentionTarget,
                                                     dOutput, 2, 3, 1, 1, 257, 0.2F, workspace, workspaceSize,
                                                     runtime.Stream()) == ACL_ERROR_INVALID_PARAM;
    AttentionExample::Check(
        aclnnCsrGatAttentionAggregateFused(dRow, dSource, dProjected, dAttentionSource, dAttentionTarget, dOutput, 2, 3,
                                           1, 1, 2, 0.2F, workspace, workspaceSize, runtime.Stream()),
        "launch");
    AttentionExample::Check(aclrtSynchronizeStream(runtime.Stream()), "synchronize");
    AttentionExample::CopyOutput(&output, dOutput);
    passed = passed && AttentionExample::Matches(output, expected, 1.0e-5F);
    if (passed)
    {
        std::puts("PASSED");
    }
    return passed ? 0 : 1;
}
