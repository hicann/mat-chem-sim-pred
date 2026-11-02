/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <array>

#include "../../acl_smoke_common.h"
#include "csr_hypergraph_attention_two_stage_propagate_fused_host.h"

namespace
{
int RunSmoke(AclSmoke::Session& session)
{
    const std::vector<int32_t> edgeRow{0, 2, 4}, nodeIndex{0, 1, 1, 2};
    const std::vector<int32_t> nodeRow{0, 1, 3, 4}, edgeIndex{0, 0, 1, 1}, position{0, 1, 2, 3};
    const std::vector<float> edgeScale{0.5F, 0.5F}, nodeScale{1.0F, 0.5F, 1.0F};
    const std::vector<float> features{2.0F, 4.0F, 6.0F}, logits(4, 0.0F), expected{0.75F, 1.0F, 1.25F};
    std::vector<float> output(3, 0.0F);
    AclSmoke::Buffer edgeRowData(edgeRow.size() * sizeof(int32_t));
    AclSmoke::Buffer nodeIndexData(nodeIndex.size() * sizeof(int32_t));
    AclSmoke::Buffer edgeScaleData(edgeScale.size() * sizeof(float));
    AclSmoke::Buffer nodeRowData(nodeRow.size() * sizeof(int32_t)),
        edgeIndexData(edgeIndex.size() * sizeof(int32_t));
    AclSmoke::Buffer positionData(position.size() * sizeof(int32_t)),
        nodeScaleData(nodeScale.size() * sizeof(float));
    AclSmoke::Buffer featureData(features.size() * sizeof(float)), logitsData(logits.size() * sizeof(float));
    AclSmoke::Buffer outputData(output.size() * sizeof(float));
    const uint64_t bytes = aclnnCsrHypergraphAttentionTwoStagePropagateFusedGetWorkspaceSize(3, 2, 4, 1, 1, 2, 2);
    AclSmoke::Buffer workspace(bytes);
    edgeRowData.CopyFrom(edgeRow);
    nodeIndexData.CopyFrom(nodeIndex);
    edgeScaleData.CopyFrom(edgeScale);
    nodeRowData.CopyFrom(nodeRow);
    edgeIndexData.CopyFrom(edgeIndex);
    positionData.CopyFrom(position);
    nodeScaleData.CopyFrom(nodeScale);
    featureData.CopyFrom(features);
    logitsData.CopyFrom(logits);
    AclSmoke::Check(
        aclnnCsrHypergraphAttentionTwoStagePropagateFused(
            edgeRowData.Data(), nodeIndexData.Data(), edgeScaleData.Data(), nodeRowData.Data(), edgeIndexData.Data(),
            positionData.Data(), nodeScaleData.Data(), featureData.Data(), logitsData.Data(), outputData.Data(), 3, 2,
            4, 1, 1, 2, 2, 0, 0.2F, workspace.Data(), workspace.Size(), session.Stream()),
        "launch");
    AclSmoke::Check(aclrtSynchronizeStream(session.Stream()), "synchronize");
    outputData.CopyTo(output);
    return AclSmoke::Validate(output, expected);
}
}  // namespace

int main(int argc, char** argv)
{
    AclSmoke::Session session(argc, argv);
    return RunSmoke(session);
}
