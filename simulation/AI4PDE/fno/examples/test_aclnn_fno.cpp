#include "acl/acl.h"
#include <iostream>
#include <vector>
#include <cmath>

extern "C" int32_t aclnnFNOForward(
    void* inputAddr, void* weightsAddr, void* outputAddr,
    int32_t batchSize, int32_t gridDim,
    int32_t inChannels, int32_t outChannels,
    int32_t hiddenChannels, int32_t modes,
    void* workspaceAddr, uint64_t workspaceSize, void* stream);

extern "C" uint64_t aclnnFNOForwardGetWorkspaceSize(
    int32_t batchSize, int32_t gridDim,
    int32_t inChannels, int32_t outChannels,
    int32_t hiddenChannels, int32_t modes);

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    const int32_t batchSize = 1, gridDim = 8;
    const int32_t inCh = 1, hiddenCh = 2, outCh = 1, modes = 2;

    int32_t totalW = (inCh * hiddenCh + hiddenCh) +
                     (modes * hiddenCh * hiddenCh * 2) +
                     (hiddenCh * outCh + outCh);

    std::vector<float> hInput(batchSize * gridDim * inCh, 1.0f);
    std::vector<float> hWeights(totalW, 0.0f);
    std::vector<float> hOutput(batchSize * gridDim * outCh, 0.0f);

    uint64_t wsSize = aclnnFNOForwardGetWorkspaceSize(
        batchSize, gridDim, inCh, outCh, hiddenCh, modes);

    void *dInput, *dWeights, *dOutput, *dWorkspace;
    aclrtMalloc(&dInput, hInput.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dWeights, hWeights.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dOutput, hOutput.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dWorkspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpyAsync(dInput, hInput.size() * sizeof(float), hInput.data(),
                      hInput.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(dWeights, hWeights.size() * sizeof(float), hWeights.data(),
                      hWeights.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);

    aclnnFNOForward(dInput, dWeights, dOutput,
                     batchSize, gridDim, inCh, outCh, hiddenCh, modes,
                     dWorkspace, wsSize, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hOutput.data(), hOutput.size() * sizeof(float),
                 dOutput, hOutput.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "FNO demo: output[0] = " << hOutput[0] << std::endl;
    std::cout << "DONE" << std::endl;

    aclrtFree(dInput); aclrtFree(dWeights);
    aclrtFree(dOutput); aclrtFree(dWorkspace);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
