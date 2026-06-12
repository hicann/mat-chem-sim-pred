/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include <iostream>
#include <vector>
#include <cmath>

extern "C" int32_t aclnnMeshGraphNet(
    void* nodeFeaturesAddr, void* edgeIndicesAddr,
    void* edgeFeaturesAddr, void* nodeWeightsAddr,
    void* edgeWeightsAddr, void* outputAddr,
    int32_t numNodes, int32_t numEdges,
    int32_t nodeDim, int32_t edgeDim,
    int32_t hiddenDim, int32_t outputDim,
    int32_t maxNeighbors,
    void* workspaceAddr, uint64_t workspaceSize, void* stream);

extern "C" uint64_t aclnnMeshGraphNetGetWorkspaceSize(
    int32_t numNodes, int32_t numEdges,
    int32_t nodeDim, int32_t edgeDim,
    int32_t hiddenDim, int32_t outputDim);

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    const int32_t numNodes = 5, numEdges = 4;
    const int32_t nodeDim = 3, edgeDim = 2, hiddenDim = 8, outputDim = 2;
    const int32_t maxNeighbors = 4;

    std::vector<float> hNodeFeat(numNodes * nodeDim, 1.0f);
    std::vector<int32_t> hEdgeIdx = {0, 1, 1, 2, 2, 3, 3, 4};
    std::vector<float> hEdgeFeat(numEdges * edgeDim, 0.5f);

    int32_t nwSize = (nodeDim * hiddenDim + hiddenDim) +
                     (hiddenDim * hiddenDim + hiddenDim) +
                     (hiddenDim * outputDim + outputDim);
    int32_t ewSize = ((nodeDim + edgeDim) * hiddenDim + hiddenDim) +
                     (hiddenDim * hiddenDim + hiddenDim);

    std::vector<float> hNodeW(nwSize, 0.1f);
    std::vector<float> hEdgeW(ewSize, 0.1f);
    std::vector<float> hOutput(numNodes * outputDim, 0.0f);

    uint64_t wsSize = aclnnMeshGraphNetGetWorkspaceSize(
        numNodes, numEdges, nodeDim, edgeDim, hiddenDim, outputDim);

    void *dNF, *dEI, *dEF, *dNW, *dEW, *dOut, *dWS;
    aclrtMalloc(&dNF, hNodeFeat.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dEI, hEdgeIdx.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dEF, hEdgeFeat.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dNW, hNodeW.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dEW, hEdgeW.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dOut, hOutput.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dWS, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpyAsync(dNF, hNodeFeat.size() * sizeof(float), hNodeFeat.data(),
                      hNodeFeat.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(dEI, hEdgeIdx.size() * sizeof(int32_t), hEdgeIdx.data(),
                      hEdgeIdx.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(dEF, hEdgeFeat.size() * sizeof(float), hEdgeFeat.data(),
                      hEdgeFeat.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(dNW, hNodeW.size() * sizeof(float), hNodeW.data(),
                      hNodeW.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(dEW, hEdgeW.size() * sizeof(float), hEdgeW.data(),
                      hEdgeW.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);

    aclnnMeshGraphNet(dNF, dEI, dEF, dNW, dEW, dOut,
                       numNodes, numEdges, nodeDim, edgeDim,
                       hiddenDim, outputDim, maxNeighbors,
                       dWS, wsSize, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hOutput.data(), hOutput.size() * sizeof(float),
                 dOut, hOutput.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "MeshGraphNet demo: output[0] = " << hOutput[0]
              << ", output[4] = " << hOutput[4] << std::endl;
    std::cout << "DONE" << std::endl;

    aclrtFree(dNF); aclrtFree(dEI); aclrtFree(dEF);
    aclrtFree(dNW); aclrtFree(dEW); aclrtFree(dOut); aclrtFree(dWS);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
