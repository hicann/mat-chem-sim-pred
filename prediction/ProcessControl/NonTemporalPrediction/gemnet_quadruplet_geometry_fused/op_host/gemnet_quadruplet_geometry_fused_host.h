/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#pragma once
#include <cstdint>
extern "C" uint64_t aclnnGemNetQuadrupletGeometryFusedGetWorkspaceSize(int64_t nodes, int64_t edges,
                                                                       int64_t interactionEdges, int64_t quadruplets,
                                                                       int64_t intermediateCa, int64_t intermediateDb);
extern "C" int32_t aclnnGemNetQuadrupletGeometryFused(
    void* position, void* sourceIndex, void* targetIndex, void* interactionSource, void* interactionTarget,
    void* reduceCa, void* expandDb, void* reduceIntermediateCa, void* expandIntermediateDb, void* reduceIntermediateAb,
    void* expandIntermediateAb, void* angleCab, void* angleAbd, void* thetaCabd, int64_t nodes, int64_t edges,
    int64_t interactionEdges, int64_t quadruplets, int64_t intermediateCa, int64_t intermediateDb, void* workspace,
    uint64_t workspaceSize, void* stream);
