/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MESH_GRAPH_NET_KERNEL_H
#define MESH_GRAPH_NET_KERNEL_H

#include "kernel_operator.h"
#include "pde_math.h"

using namespace AscendC;

struct MeshGraphNetTilingData {
    int32_t numNodes;
    int32_t numEdges;
    int32_t nodeDim;
    int32_t edgeDim;
    int32_t hiddenDim;
    int32_t outputDim;
    int32_t maxNeighbors;
    int32_t tileSize;
    int32_t coreNum;
    int32_t numEdgeTiles;
    int32_t nodeWeightSize;
    int32_t edgeWeightSize;
};

constexpr int32_t MGN_MAX_NODES = 4096;
constexpr int32_t MGN_MAX_EDGES = 32768;
constexpr int32_t MGN_MAX_HIDDEN = 128;
constexpr int32_t MGN_MAX_NEIGHBORS = 64;
constexpr int32_t MGN_BLOCK_SIZE = 32;
constexpr int32_t MGN_MAX_CORES = 32;

__aicore__ inline uint32_t MgnAlignUp(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

class MeshGraphNetOp {
public:
    __aicore__ inline MeshGraphNetOp() {}

    __aicore__ inline void Init(GM_ADDR nodeFeatures, GM_ADDR edgeIndices,
                                 GM_ADDR edgeFeatures, GM_ADDR nodeWeights,
                                 GM_ADDR edgeWeights, GM_ADDR output,
                                 GM_ADDR tiling) {
        __gm__ const MeshGraphNetTilingData* td =
            reinterpret_cast<__gm__ const MeshGraphNetTilingData*>(tiling);

        numNodes_ = td->numNodes;
        numEdges_ = td->numEdges;
        nodeDim_ = td->nodeDim;
        edgeDim_ = td->edgeDim;
        hiddenDim_ = td->hiddenDim;
        outputDim_ = td->outputDim;
        maxNeighbors_ = td->maxNeighbors;
        tileSize_ = td->tileSize;
        coreNum_ = td->coreNum;

        coreIdx_ = GetBlockIdx();

        int32_t nodesPerCore = (numNodes_ + coreNum_ - 1) / coreNum_;
        myStart_ = coreIdx_ * nodesPerCore;
        myEnd_ = myStart_ + nodesPerCore;
        if (myEnd_ > numNodes_) myEnd_ = numNodes_;
        if (myStart_ >= numNodes_) {
            myStart_ = 0;
            myEnd_ = 0;
        }
        myCount_ = myEnd_ - myStart_;

        nodeFeatures_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(nodeFeatures));
        edgeIndices_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(edgeIndices));
        edgeFeatures_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(edgeFeatures));
        nodeWeights_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(nodeWeights));
        edgeWeights_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(edgeWeights));
        output_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));

        nodeWOffset_ = 0;
        nodeW2Offset_ = nodeDim_ * hiddenDim_ + hiddenDim_;
        edgeWOffset_ = 0;
        edgeW2Offset_ = (nodeDim_ + edgeDim_) * hiddenDim_ + hiddenDim_;
    }

    __aicore__ inline void MLP2Layer(const float* input, float* output,
                                      const GlobalTensor<float>& weights,
                                      int32_t wOff, int32_t inDim, int32_t hiddenDim, int32_t outDim) {
        float h[MGN_MAX_HIDDEN];

        for (int32_t j = 0; j < hiddenDim; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < inDim; k++)
                sum += input[k] * weights.GetValue(wOff + j * inDim + k);
            float bias = weights.GetValue(wOff + inDim * hiddenDim + j);
            h[j] = pde_math::pde_reluf(sum + bias);
        }

        int32_t wOff2 = wOff + inDim * hiddenDim + hiddenDim;
        for (int32_t j = 0; j < outDim; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < hiddenDim; k++)
                sum += h[k] * weights.GetValue(wOff2 + j * hiddenDim + k);
            float bias = weights.GetValue(wOff2 + hiddenDim * outDim + j);
            output[j] = sum + bias;
        }
    }

    __aicore__ inline void Process() {
        if (myCount_ <= 0) return;

        float nodeH[MGN_MAX_HIDDEN];
        float edgeAggr[MGN_MAX_HIDDEN];
        float edgeConcat[MGN_MAX_HIDDEN * 2];

        for (int32_t i = 0; i < myCount_; i++) {
            int32_t nodeIdx = myStart_ + i;

            for (int32_t d = 0; d < hiddenDim_; d++) {
                edgeAggr[d] = 0.0f;
            }

            int32_t neighborCount = 0;
            for (int32_t e = 0; e < numEdges_; e++) {
                int32_t src = edgeIndices_gm_.GetValue(e * 2);
                int32_t dst = edgeIndices_gm_.GetValue(e * 2 + 1);
                if (dst != nodeIdx) continue;
                if (neighborCount >= MGN_MAX_NEIGHBORS) break;

                float nodeFeat[MGN_MAX_HIDDEN];
                for (int32_t d = 0; d < nodeDim_; d++) {
                    nodeFeat[d] = nodeFeatures_gm_.GetValue(src * nodeDim_ + d);
                }

                float edgeFeat[MGN_MAX_HIDDEN];
                for (int32_t d = 0; d < edgeDim_; d++) {
                    edgeFeat[d] = edgeFeatures_gm_.GetValue(e * edgeDim_ + d);
                }

                for (int32_t d = 0; d < nodeDim_; d++) {
                    edgeConcat[d] = nodeFeat[d];
                }
                for (int32_t d = 0; d < edgeDim_; d++) {
                    edgeConcat[nodeDim_ + d] = edgeFeat[d];
                }

                float edgeMsg[MGN_MAX_HIDDEN];
                MLP2Layer(edgeConcat, edgeMsg, edgeWeights_gm_,
                           edgeWOffset_, nodeDim_ + edgeDim_,
                           hiddenDim_, hiddenDim_);

                for (int32_t d = 0; d < hiddenDim_; d++) {
                    edgeAggr[d] += edgeMsg[d];
                }
                neighborCount++;
            }

            if (neighborCount > 0) {
                float invN = 1.0f / neighborCount;
                for (int32_t d = 0; d < hiddenDim_; d++) {
                    edgeAggr[d] *= invN;
                }
            }

            float selfFeat[MGN_MAX_HIDDEN];
            for (int32_t d = 0; d < nodeDim_; d++) {
                selfFeat[d] = nodeFeatures_gm_.GetValue(nodeIdx * nodeDim_ + d);
            }

            float nodeUpdate[MGN_MAX_HIDDEN];
            MLP2Layer(selfFeat, nodeUpdate, nodeWeights_gm_,
                       nodeWOffset_, nodeDim_, hiddenDim_, hiddenDim_);

            for (int32_t d = 0; d < hiddenDim_; d++) {
                nodeH[d] = nodeUpdate[d] + edgeAggr[d];
            }

            float finalOut[MGN_MAX_HIDDEN];
            MLP2Layer(nodeH, finalOut, nodeWeights_gm_,
                       nodeW2Offset_, hiddenDim_, hiddenDim_, outputDim_);

            for (int32_t d = 0; d < outputDim_; d++) {
                output_gm_.SetValue(nodeIdx * outputDim_ + d, finalOut[d]);
            }
        }
    }

private:
    GlobalTensor<float> nodeFeatures_gm_;
    GlobalTensor<int32_t> edgeIndices_gm_;
    GlobalTensor<float> edgeFeatures_gm_;
    GlobalTensor<float> nodeWeights_gm_;
    GlobalTensor<float> edgeWeights_gm_;
    GlobalTensor<float> output_gm_;

    TPipe pipe_;

    int32_t numNodes_;
    int32_t numEdges_;
    int32_t nodeDim_;
    int32_t edgeDim_;
    int32_t hiddenDim_;
    int32_t outputDim_;
    int32_t maxNeighbors_;
    int32_t tileSize_;
    int32_t coreNum_;
    int32_t coreIdx_;
    int32_t myStart_;
    int32_t myEnd_;
    int32_t myCount_;

    int32_t nodeWOffset_;
    int32_t nodeW2Offset_;
    int32_t edgeWOffset_;
    int32_t edgeW2Offset_;
};

#endif
