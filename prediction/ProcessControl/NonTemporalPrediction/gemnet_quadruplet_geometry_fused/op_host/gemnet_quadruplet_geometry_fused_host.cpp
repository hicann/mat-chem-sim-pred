/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "gemnet_quadruplet_geometry_fused_host.h"

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"

extern "C" uint32_t aclrtlaunch_gemnet_quadruplet_geometry_fused_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                        void*, void*, void*, void*, void*, void*, void*,
                                                                        void*, void*, void*, void*, uint32_t, uint32_t,
                                                                        uint32_t, uint32_t, uint32_t, uint32_t);

extern "C" uint64_t aclnnGemNetQuadrupletGeometryFusedGetWorkspaceSize(int64_t nodes, int64_t edges,
                                                                       int64_t interactionEdges, int64_t quadruplets,
                                                                       int64_t intermediateCa, int64_t intermediateDb)
{
    (void)nodes;
    (void)edges;
    (void)interactionEdges;
    (void)quadruplets;
    (void)intermediateCa;
    (void)intermediateDb;
    return 0U;
}

namespace
{
bool ValidGeometryInputs(void* const* pointers, int64_t nodes, int64_t edges, int64_t interactionEdges,
                         int64_t quadruplets, int64_t intermediateCa, int64_t intermediateDb, void* workspace,
                         uint64_t workspaceSize, void* stream)
{
    const bool sizes = nodes > 0 && nodes <= INT32_MAX && edges > 0 && edges <= INT32_MAX && interactionEdges > 0 &&
                       interactionEdges <= INT32_MAX && quadruplets > 0 && quadruplets <= INT32_MAX &&
                       intermediateCa > 0 && intermediateCa <= INT32_MAX && intermediateDb > 0 &&
                       intermediateDb <= INT32_MAX;
    if (!sizes || workspace != nullptr || workspaceSize != 0U || stream == nullptr)
    {
        return false;
    }
    for (uint32_t index = 0U; index < 14U; ++index)
    {
        if (pointers[index] == nullptr)
        {
            return false;
        }
    }
    return true;
}
}  // namespace

extern "C" int32_t aclnnGemNetQuadrupletGeometryFused(
    void* position, void* sourceIndex, void* targetIndex, void* interactionSource, void* interactionTarget,
    void* reduceCa, void* expandDb, void* reduceIntermediateCa, void* expandIntermediateDb, void* reduceIntermediateAb,
    void* expandIntermediateAb, void* angleCab, void* angleAbd, void* thetaCabd, int64_t nodes, int64_t edges,
    int64_t interactionEdges, int64_t quadruplets, int64_t intermediateCa, int64_t intermediateDb, void* workspace,
    uint64_t workspaceSize, void* stream)
{
    void* pointers[] = {position,
                        sourceIndex,
                        targetIndex,
                        interactionSource,
                        interactionTarget,
                        reduceCa,
                        expandDb,
                        reduceIntermediateCa,
                        expandIntermediateDb,
                        reduceIntermediateAb,
                        expandIntermediateAb,
                        angleCab,
                        angleAbd,
                        thetaCabd};
    if (!ValidGeometryInputs(pointers, nodes, edges, interactionEdges, quadruplets, intermediateCa, intermediateDb,
                             workspace, workspaceSize, stream))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    (void)workspace;
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(40, quadruplets)));
    return static_cast<int32_t>(aclrtlaunch_gemnet_quadruplet_geometry_fused_kernel(
        cores, reinterpret_cast<aclrtStream>(stream), position, sourceIndex, targetIndex, interactionSource,
        interactionTarget, reduceCa, expandDb, reduceIntermediateCa, expandIntermediateDb, reduceIntermediateAb,
        expandIntermediateAb, angleCab, angleAbd, thetaCabd, static_cast<uint32_t>(nodes), static_cast<uint32_t>(edges),
        static_cast<uint32_t>(interactionEdges), static_cast<uint32_t>(quadruplets),
        static_cast<uint32_t>(intermediateCa), static_cast<uint32_t>(intermediateDb)));
}
