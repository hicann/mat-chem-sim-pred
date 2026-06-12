/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "acl/acl.h"

extern "C" int32_t aclnnPinnFC(
    void* inputAddr, void* weightsAddr,
    void* outputAddr, void* gradientAddr,
    int32_t batchSize, int32_t inputDim,
    int32_t hiddenDim, int32_t outputDim,
    int32_t numLayers, int32_t activationType,
    void* workspaceAddr, uint64_t workspaceSize, void* stream);

extern "C" uint64_t aclnnPinnFCGetWorkspaceSize(
    int32_t batchSize, int32_t inputDim,
    int32_t hiddenDim, int32_t outputDim,
    int32_t numLayers);

static float Tanh(float x) { return tanhf(x); }
static float TanhDeriv(float x) { float t = tanhf(x); return 1.0f - t * t; }

void CpuFCWithGrad(const float* input, const float* weights, float* output, float* gradient,
                    int32_t batchSize, int32_t inputDim,
                    int32_t hiddenDim, int32_t outputDim, int32_t numLayers) {
    int32_t dims[8], wOffsets[8];
    dims[0] = inputDim;
    for (int l = 1; l < numLayers; l++) dims[l] = hiddenDim;
    dims[numLayers] = outputDim;
    wOffsets[0] = 0;
    for (int l = 0; l < numLayers; l++)
        wOffsets[l+1] = wOffsets[l] + dims[l] * dims[l+1] + dims[l+1];

    for (int b = 0; b < batchSize; b++) {
        float actv[8][256], pre[8][256];
        for (int d = 0; d < inputDim; d++) actv[0][d] = input[b * inputDim + d];

        for (int l = 0; l < numLayers; l++) {
            int inD = dims[l], outD = dims[l+1], wOff = wOffsets[l];
            for (int j = 0; j < outD; j++) {
                float sum = 0;
                for (int k = 0; k < inD; k++)
                    sum += actv[l][k] * weights[wOff + j * inD + k];
                float bias = weights[wOff + inD * outD + j];
                pre[l][j] = sum + bias;
                actv[l+1][j] = Tanh(sum + bias);
            }
        }

        for (int o = 0; o < outputDim; o++)
            output[b * outputDim + o] = actv[numLayers][o];

        for (int o = 0; o < outputDim; o++) {
            float grad[256];
            for (int d = 0; d < outputDim; d++)
                grad[numLayers] = (d == o) ? 1.0f : 0.0f;

            for (int l = numLayers - 1; l >= 0; l--) {
                int inD = dims[l], outD = dims[l+1], wOff = wOffsets[l];
                float delta[256];
                for (int j = 0; j < outD; j++)
                    delta[j] = grad[l+1] * TanhDeriv(pre[l][j]);

                for (int k = 0; k < inD; k++) {
                    float sum = 0;
                    for (int j = 0; j < outD; j++)
                        sum += weights[wOff + j * inD + k] * delta[j];
                    grad[l] = sum;
                }
            }

            for (int d = 0; d < inputDim; d++)
                gradient[(b * outputDim + o) * inputDim + d] = grad[0];
        }
    }
}

void TestSmallScale() {
    const int32_t batchSize = 8, inputDim = 3, hiddenDim = 16, outputDim = 1, numLayers = 2;

    int32_t totalW = 0, inDim = inputDim;
    for (int l = 0; l < numLayers; l++) {
        int outDim = (l == numLayers - 1) ? outputDim : hiddenDim;
        totalW += inDim * outDim + outDim;
        inDim = outDim;
    }

    std::vector<float> hInput(batchSize * inputDim);
    std::vector<float> hWeights(totalW);
    std::vector<float> hOutput(batchSize * outputDim);
    std::vector<float> hGradient(batchSize * outputDim * inputDim);
    for (auto& v : hInput) v = (rand() % 100) / 100.0f - 0.5f;
    for (auto& v : hWeights) v = (rand() % 100) / 100.0f - 0.5f;

    CpuFCWithGrad(hInput.data(), hWeights.data(), hOutput.data(), hGradient.data(),
                   batchSize, inputDim, hiddenDim, outputDim, numLayers);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint64_t wsSize = aclnnPinnFCGetWorkspaceSize(batchSize, inputDim, hiddenDim, outputDim, numLayers);
    void *dInput, *dWeights, *dOutput, *dGradient, *dWorkspace;
    aclrtMalloc(&dInput, batchSize * inputDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dWeights, totalW * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dOutput, batchSize * outputDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dGradient, batchSize * outputDim * inputDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dWorkspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpyAsync(dInput, batchSize * inputDim * sizeof(float), hInput.data(),
                      batchSize * inputDim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(dWeights, totalW * sizeof(float), hWeights.data(),
                      totalW * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);

    aclnnPinnFC(dInput, dWeights, dOutput, dGradient,
                 batchSize, inputDim, hiddenDim, outputDim,
                 numLayers, 0, dWorkspace, wsSize, stream);

    aclrtSynchronizeStream(stream);

    std::vector<float> npuOutput(batchSize * outputDim);
    std::vector<float> npuGradient(batchSize * outputDim * inputDim);
    aclrtMemcpy(npuOutput.data(), batchSize * outputDim * sizeof(float),
                 dOutput, batchSize * outputDim * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(npuGradient.data(), batchSize * outputDim * inputDim * sizeof(float),
                 dGradient, batchSize * outputDim * inputDim * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    float maxErr = 0;
    for (int i = 0; i < batchSize * outputDim; i++)
        maxErr = std::max(maxErr, std::abs(npuOutput[i] - hOutput[i]));

    float maxGradErr = 0;
    for (int i = 0; i < batchSize * outputDim * inputDim; i++)
        maxGradErr = std::max(maxGradErr, std::abs(npuGradient[i] - hGradient[i]));

    std::cout << "PINN UT: output max err = " << maxErr
              << ", gradient max err = " << maxGradErr << std::endl;
    std::cout << (maxErr < 1e-3 && maxGradErr < 1e-3 ? "PASSED" : "FAILED") << std::endl;

    aclrtFree(dInput); aclrtFree(dWeights); aclrtFree(dOutput);
    aclrtFree(dGradient); aclrtFree(dWorkspace);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

int main() {
    TestSmallScale();
    return 0;
}
