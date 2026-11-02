/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <limits>

#include "../../acl_smoke_common.h"
#include "csr_gcn2_residual_propagate_fused_host.h"

namespace
{
int RunSmoke(AclSmoke::Session& session)
{
    const std::vector<int32_t> rowPtr{0, 1, 2}, source{1, 0};
    const std::vector<float> weight{1.0F, 1.0F}, current{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> initial{5.0F, 6.0F, 7.0F, 8.0F}, expected{3.5F, 4.5F, 2.5F, 3.5F};
    std::vector<float> output(expected.size(), 0.0F);
    AclSmoke::Buffer row(rowPtr.size() * sizeof(int32_t)), sourceData(source.size() * sizeof(int32_t));
    AclSmoke::Buffer weightData(weight.size() * sizeof(float)), currentData(current.size() * sizeof(float));
    AclSmoke::Buffer initialData(initial.size() * sizeof(float)), outputData(output.size() * sizeof(float));
    AclSmoke::Buffer workspace(aclnnCsrGcn2ResidualPropagateFusedGetWorkspaceSize(2, 2, 2));
    row.CopyFrom(rowPtr);
    sourceData.CopyFrom(source);
    weightData.CopyFrom(weight);
    currentData.CopyFrom(current);
    initialData.CopyFrom(initial);
    const int64_t overflow = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    if (aclnnCsrGcn2ResidualPropagateFused(
            row.Data(), sourceData.Data(), weightData.Data(), currentData.Data(), initialData.Data(), outputData.Data(),
            overflow, 2, 2, 0.25F, workspace.Data(), workspace.Size(), session.Stream()) != ACL_ERROR_INVALID_PARAM)
    {
        return 1;
    }
    AclSmoke::Check(aclnnCsrGcn2ResidualPropagateFused(
                             row.Data(), sourceData.Data(), weightData.Data(), currentData.Data(), initialData.Data(),
                             outputData.Data(), 2, 2, 2, 0.25F, workspace.Data(), workspace.Size(), session.Stream()),
                         "launch");
    AclSmoke::Check(aclrtSynchronizeStream(session.Stream()), "sync");
    outputData.CopyTo(output);
    return AclSmoke::Validate(output, expected);
}
}  // namespace

int main(int argc, char** argv)
{
    AclSmoke::Session session(argc, argv);
    return RunSmoke(session);
}
