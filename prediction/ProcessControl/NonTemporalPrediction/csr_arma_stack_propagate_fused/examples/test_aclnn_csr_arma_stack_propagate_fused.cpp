/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "../../sparse_acl_smoke_helpers.h"
#include "csr_arma_stack_propagate_fused_host.h"

namespace
{
int RunSmoke(SparseSmoke::Session& session)
{
    const std::vector<int32_t> rowPtr{0, 1, 2}, source{1, 0};
    const std::vector<float> weight{1.0F, 0.5F};
    const std::vector<float> projected{1.0F, 2.0F, 3.0F, 4.0F, -1.0F, 2.0F, 2.0F, -3.0F};
    const std::vector<float> root{0.5F, 0.5F, 1.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.0F};
    const std::vector<float> bias{0.25F, -0.25F, 0.5F, 0.5F};
    const std::vector<float> expected{3.75F, 4.25F, 1.75F, 1.75F, 2.5F, 0.0F, 1.0F, 1.5F};
    std::vector<float> output(expected.size(), 0.0F);
    SparseSmoke::Buffer row(rowPtr.size() * sizeof(int32_t)), sourceData(source.size() * sizeof(int32_t));
    SparseSmoke::Buffer weightData(weight.size() * sizeof(float)), projectedData(projected.size() * sizeof(float));
    SparseSmoke::Buffer rootData(root.size() * sizeof(float)), biasData(bias.size() * sizeof(float));
    SparseSmoke::Buffer outputData(output.size() * sizeof(float));
    SparseSmoke::Buffer workspace(aclnnCsrArmaStackPropagateFusedGetWorkspaceSize(2, 2, 2, 2));
    row.CopyFrom(rowPtr);
    sourceData.CopyFrom(source);
    weightData.CopyFrom(weight);
    projectedData.CopyFrom(projected);
    rootData.CopyFrom(root);
    biasData.CopyFrom(bias);
    if (aclnnCsrArmaStackPropagateFused(row.Data(), sourceData.Data(), weightData.Data(), projectedData.Data(),
                                        rootData.Data(), biasData.Data(), outputData.Data(), 2, 2, 65, 2, 1,
                                        workspace.Data(), workspace.Size(),
                                        session.Stream()) != ACL_ERROR_INVALID_PARAM)
    {
        return 1;
    }
    SparseSmoke::Check(
        aclnnCsrArmaStackPropagateFused(row.Data(), sourceData.Data(), weightData.Data(), projectedData.Data(),
                                        rootData.Data(), biasData.Data(), outputData.Data(), 2, 2, 2, 2, 1,
                                        workspace.Data(), workspace.Size(), session.Stream()),
        "launch");
    SparseSmoke::Check(aclrtSynchronizeStream(session.Stream()), "sync");
    outputData.CopyTo(output);
    return SparseSmoke::Validate(output, expected);
}
}  // namespace

int main(int argc, char** argv)
{
    SparseSmoke::Session session(argc, argv);
    return RunSmoke(session);
}
