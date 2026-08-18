/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <vector>

#include "../../attention_example_common.h"
#include "csr_transformer_dot_attention_aggregate_fused_host.h"

namespace
{
struct Inputs
{
    std::vector<int32_t> row{0, 1, 2};
    std::vector<int32_t> source{1, 0};
    std::vector<float> query{1.0F, 0.0F, 0.0F, 1.0F};
    std::vector<float> key{1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> value{5.0F, 6.0F, 7.0F, 8.0F};
};
}  // namespace

int main(int argc, char** argv)
{
    int selectedDevice = 0;
    if (!AttentionExample::ParseDevice(argc, argv, &selectedDevice))
    {
        return 1;
    }
    AttentionExample::Runtime context(selectedDevice);
    AttentionExample::Buffers allocations;
    const Inputs input;
    const std::vector<float> expected{7.0F, 8.0F, 5.0F, 6.0F};
    std::vector<float> output(expected.size(), 0.0F);
    void* row = allocations.Copy(input.row);
    void* source = allocations.Copy(input.source);
    void* query = allocations.Copy(input.query);
    void* key = allocations.Copy(input.key);
    void* value = allocations.Copy(input.value);
    void* result = allocations.Allocate(output.size() * sizeof(float));
    const uint64_t workspaceSize = aclnnCsrTransformerDotAttentionAggregateFusedGetWorkspaceSize(2, 2, 1, 2, 1);
    void* workspace = allocations.Allocate(workspaceSize);
    bool passed = aclnnCsrTransformerDotAttentionAggregateFused(row, source, query, key, value, result, 2, 2, 1, 2, 257,
                                                                workspace, workspaceSize,
                                                                context.Stream()) == ACL_ERROR_INVALID_PARAM;
    AttentionExample::Check(
        aclnnCsrTransformerDotAttentionAggregateFused(row, source, query, key, value, result, 2, 2, 1, 2, 1, workspace,
                                                      workspaceSize, context.Stream()),
        "Transformer attention launch");
    AttentionExample::Check(aclrtSynchronizeStream(context.Stream()), "Transformer synchronize");
    AttentionExample::CopyOutput(&output, result);
    passed = passed && AttentionExample::Matches(output, expected, 1.0e-4F);
    return passed ? 0 : 1;
}
