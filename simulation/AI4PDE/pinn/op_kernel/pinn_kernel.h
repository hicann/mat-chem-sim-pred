/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PINN_KERNEL_H
#define PINN_KERNEL_H

#include "kernel_operator.h"
#include "pde_math.h"

using namespace AscendC;

struct PinnTilingData {
    int32_t batchSize;
    int32_t inputDim;
    int32_t hiddenDim;
    int32_t outputDim;
    int32_t numLayers;
    int32_t tileSize;
    int32_t coreNum;
    int32_t weightsPerLayer;
    int32_t totalWeightSize;
    int32_t activationType;
};

constexpr int32_t PINN_MAX_LAYERS = 8;
constexpr int32_t PINN_MAX_HIDDEN = 256;
constexpr int32_t PINN_BLOCK_SIZE = 32;
constexpr int32_t PINN_MAX_CORES = 32;

__aicore__ inline uint32_t PinnAlignUp(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

class PinnFCOp {
public:
    __aicore__ inline PinnFCOp() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weights, GM_ADDR output,
                                 GM_ADDR gradient, GM_ADDR tiling) {
        __gm__ const PinnTilingData* td =
            reinterpret_cast<__gm__ const PinnTilingData*>(tiling);

        batchSize_ = td->batchSize;
        inputDim_ = td->inputDim;
        hiddenDim_ = td->hiddenDim;
        outputDim_ = td->outputDim;
        numLayers_ = td->numLayers;
        tileSize_ = td->tileSize;
        coreNum_ = td->coreNum;
        activationType_ = td->activationType;

        coreIdx_ = GetBlockIdx();

        int32_t ptsPerCore = (batchSize_ + coreNum_ - 1) / coreNum_;
        myStart_ = coreIdx_ * ptsPerCore;
        myEnd_ = myStart_ + ptsPerCore;
        if (myEnd_ > batchSize_) myEnd_ = batchSize_;
        if (myStart_ >= batchSize_) {
            myStart_ = 0;
            myEnd_ = 0;
        }
        myCount_ = myEnd_ - myStart_;

        input_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
        weights_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
        output_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        gradient_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradient));

        uint32_t tileBufSize = PinnAlignUp(tileSize_ * inputDim_ * sizeof(float), PINN_BLOCK_SIZE);
        uint32_t hiddenBufSize = PinnAlignUp(PINN_MAX_HIDDEN * sizeof(float) * 2, PINN_BLOCK_SIZE);
        uint32_t outBufSize = PinnAlignUp(tileSize_ * outputDim_ * sizeof(float), PINN_BLOCK_SIZE);
        uint32_t gradBufSize = PinnAlignUp(tileSize_ * inputDim_ * outputDim_ * sizeof(float), PINN_BLOCK_SIZE);

        pipe_.InitBuffer(inputQueue_, 1, tileBufSize);
        pipe_.InitBuffer(hiddenQueue_, 2, hiddenBufSize);
        pipe_.InitBuffer(outputQueue_, 1, outBufSize);
        pipe_.InitBuffer(gradQueue_, 1, gradBufSize);

        ComputeLayerDims();
    }

    __aicore__ inline void ComputeLayerDims() {
        layerDims_[0] = inputDim_;
        for (int32_t l = 1; l < numLayers_; l++) {
            layerDims_[l] = hiddenDim_;
        }
        layerDims_[numLayers_] = outputDim_;

        weightOffsets_[0] = 0;
        for (int32_t l = 0; l < numLayers_; l++) {
            int32_t inD = layerDims_[l];
            int32_t outD = layerDims_[l + 1];
            weightOffsets_[l + 1] = weightOffsets_[l] + inD * outD + outD;
        }
    }

    __aicore__ inline float Activate(float x) {
        if (activationType_ == 0) return pde_math::pde_tanhf(x);
        if (activationType_ == 1) return pde_math::pde_sigmoidf(x);
        return pde_math::pde_reluf(x);
    }

    __aicore__ inline float ActivateDeriv(float x) {
        if (activationType_ == 0) {
            float t = pde_math::pde_tanhf(x);
            return 1.0f - t * t;
        }
        if (activationType_ == 1) {
            float s = pde_math::pde_sigmoidf(x);
            return s * (1.0f - s);
        }
        return (x > 0.0f) ? 1.0f : 0.0f;
    }

    __aicore__ inline void ForwardSingle(float* h, int32_t layerIn, int32_t layerOut,
                                           float* inputVals, int32_t weightOffset) {
        for (int32_t j = 0; j < layerOut; j++) {
            float sum = 0.0f;
            int32_t wBase = weightOffset + j * layerIn;
            for (int32_t k = 0; k < layerIn; k++) {
                sum += inputVals[k] * weights_gm_.GetValue(wBase + k);
            }
            float bias = weights_gm_.GetValue(weightOffset + layerIn * layerOut + j);
            h[j] = Activate(sum + bias);
        }
    }

    __aicore__ inline void ForwardLayer(float* h_pre, float* h_post,
                                         int32_t layerIn, int32_t layerOut,
                                         int32_t weightOffset) {
        for (int32_t j = 0; j < layerOut; j++) {
            float sum = 0.0f;
            int32_t wBase = weightOffset + j * layerIn;
            for (int32_t k = 0; k < layerIn; k++) {
                sum += h_pre[k] * weights_gm_.GetValue(wBase + k);
            }
            float bias = weights_gm_.GetValue(weightOffset + layerIn * layerOut + j);
            h_post[j] = Activate(sum + bias);
        }
    }

    __aicore__ inline void Process() {
        if (myCount_ <= 0) return;

        int32_t numTiles = (myCount_ + tileSize_ - 1) / tileSize_;

        LocalTensor<float> outputLocal = outputQueue_.AllocTensor<float>();
        LocalTensor<float> gradLocal = gradQueue_.AllocTensor<float>();
        Duplicate(gradLocal, 0.0f, myCount_ * inputDim_ * outputDim_);

        for (int32_t tile = 0; tile < numTiles; tile++) {
            int32_t tStart = tile * tileSize_;
            int32_t tEnd = tStart + tileSize_;
            if (tEnd > myCount_) tEnd = myCount_;
            int32_t tCount = tEnd - tStart;

            int32_t batchIdx = myStart_ + tStart;

            LocalTensor<float> tileInput = inputQueue_.AllocTensor<float>();
            DataCopy(tileInput, input_gm_[batchIdx * inputDim_], tCount * inputDim_);
            inputQueue_.EnQue(tileInput);
            tileInput = inputQueue_.DeQue<float>();

            for (int32_t i = 0; i < tCount; i++) {
                float x0[PINN_MAX_HIDDEN];
                for (int32_t d = 0; d < inputDim_; d++) {
                    x0[d] = tileInput.GetValue(i * inputDim_ + d);
                }

                float activations[PINN_MAX_LAYERS + 1][PINN_MAX_HIDDEN];
                float preAct[PINN_MAX_LAYERS][PINN_MAX_HIDDEN];

                for (int32_t d = 0; d < inputDim_; d++) {
                    activations[0][d] = x0[d];
                }

                for (int32_t l = 0; l < numLayers_; l++) {
                    int32_t inD = layerDims_[l];
                    int32_t outD = layerDims_[l + 1];
                    int32_t wOff = weightOffsets_[l];

                    for (int32_t j = 0; j < outD; j++) {
                        float sum = 0.0f;
                        int32_t wBase = wOff + j * inD;
                        for (int32_t k = 0; k < inD; k++) {
                            sum += activations[l][k] * weights_gm_.GetValue(wBase + k);
                        }
                        float bias = weights_gm_.GetValue(wOff + inD * outD + j);
                        preAct[l][j] = sum + bias;
                        activations[l + 1][j] = Activate(sum + bias);
                    }
                }

                for (int32_t o = 0; o < outputDim_; o++) {
                    output_gm_.SetValue((batchIdx + i) * outputDim_ + o,
                                        activations[numLayers_][o]);
                }

                for (int32_t o = 0; o < outputDim_; o++) {
                    float grad[PINN_MAX_HIDDEN];
                    for (int32_t d = 0; d < outputDim_; d++) {
                        float delta = (d == o) ? 1.0f : 0.0f;
                        grad[numLayers_] = delta;
                    }

                    for (int32_t l = numLayers_ - 1; l >= 0; l--) {
                        int32_t inD = layerDims_[l];
                        int32_t outD = layerDims_[l + 1];
                        int32_t wOff = weightOffsets_[l];

                        float delta[PINN_MAX_HIDDEN];
                        for (int32_t j = 0; j < outD; j++) {
                            delta[j] = grad[l + 1] * ActivateDeriv(preAct[l][j]);
                        }

                        for (int32_t k = 0; k < inD; k++) {
                            float sum = 0.0f;
                            int32_t wBase = wOff;
                            for (int32_t j = 0; j < outD; j++) {
                                sum += weights_gm_.GetValue(wBase + j * inD + k) * delta[j];
                            }
                            grad[l] = sum;
                        }
                    }

                    for (int32_t d = 0; d < inputDim_; d++) {
                        int32_t gradIdx = ((batchIdx + i) * outputDim_ + o) * inputDim_ + d;
                        gradient_gm_.SetValue(gradIdx, grad[0]);
                    }
                }
            }

            inputQueue_.FreeTensor(tileInput);
        }

        outputQueue_.FreeTensor(outputLocal);
        gradQueue_.FreeTensor(gradLocal);
    }

private:
    GlobalTensor<float> input_gm_;
    GlobalTensor<float> weights_gm_;
    GlobalTensor<float> output_gm_;
    GlobalTensor<float> gradient_gm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 2> hiddenQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TQue<QuePosition::VECOUT, 1> gradQueue_;

    int32_t batchSize_;
    int32_t inputDim_;
    int32_t hiddenDim_;
    int32_t outputDim_;
    int32_t numLayers_;
    int32_t tileSize_;
    int32_t coreNum_;
    int32_t activationType_;
    int32_t coreIdx_;
    int32_t myStart_;
    int32_t myEnd_;
    int32_t myCount_;

    int32_t layerDims_[PINN_MAX_LAYERS + 1];
    int32_t weightOffsets_[PINN_MAX_LAYERS + 1];
};

#endif
