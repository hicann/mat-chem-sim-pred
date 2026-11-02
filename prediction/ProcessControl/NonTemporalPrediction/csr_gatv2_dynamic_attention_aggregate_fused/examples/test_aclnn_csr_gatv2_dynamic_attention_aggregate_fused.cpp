/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <vector>

#include "../../attention_example_common.h"
#include "csr_gatv2_dynamic_attention_aggregate_fused_host.h"

int main(int argc, char** argv)
{
    int deviceId = 0;
    if (!AttentionExample::ParseDevice(argc, argv, &deviceId))
    {
        return 1;
    }
    AttentionExample::Runtime acl(deviceId);
    AttentionExample::Buffers memory;
    const std::vector<int32_t> row{0, 1, 2};
    const std::vector<int32_t> source{1, 0};
    const std::vector<float> left{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> right{0.5F, -0.5F, 1.0F, 1.0F};
    const std::vector<float> attention{0.75F, -0.25F};
    const std::vector<float> expected{3.0F, 4.0F, 1.0F, 2.0F};
    std::vector<float> actual(expected.size(), 0.0F);
    void* rowDevice = memory.Copy(row);
    void* sourceDevice = memory.Copy(source);
    void* leftDevice = memory.Copy(left);
    void* rightDevice = memory.Copy(right);
    void* attentionDevice = memory.Copy(attention);
    void* outputDevice = memory.Allocate(actual.size() * sizeof(float));
    const uint64_t bytes = aclnnCsrGatv2DynamicAttentionAggregateFusedGetWorkspaceSize(2, 2, 1, 2, 1);
    void* workspace = memory.Allocate(bytes);
    bool passed = aclnnCsrGatv2DynamicAttentionAggregateFused(
                      rowDevice, sourceDevice, leftDevice, rightDevice, attentionDevice, outputDevice, 2, 2, 1, 2, 1,
                      1.5F, workspace, bytes, acl.Stream()) == ACL_ERROR_INVALID_PARAM;
    const int32_t launchResult =
        aclnnCsrGatv2DynamicAttentionAggregateFused(rowDevice, sourceDevice, leftDevice, rightDevice, attentionDevice,
                                                    outputDevice, 2, 2, 1, 2, 1, 0.2F, workspace, bytes, acl.Stream());
    AttentionExample::Check(launchResult, "GATv2 launch");
    AttentionExample::Check(aclrtSynchronizeStream(acl.Stream()), "GATv2 synchronize");
    AttentionExample::CopyOutput(&actual, outputDevice);
    passed = passed && AttentionExample::Matches(actual, expected, 1.0e-4F);
    return passed ? 0 : 1;
}
