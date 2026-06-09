#include "acl/acl.h"
#include <iostream>
#include <vector>
#include <cmath>

extern "C" int32_t aclnnDeepOnetForward(
    void* branchInputAddr, void* trunkInputAddr,
    void* branchWeightsAddr, void* trunkWeightsAddr,
    void* outputAddr,
    int32_t batchSize, int32_t branchDim,
    int32_t trunkDim, int32_t latentDim,
    int32_t querySize,
    void* workspaceAddr, uint64_t workspaceSize, void* stream);

extern "C" uint64_t aclnnDeepOnetForwardGetWorkspaceSize(
    int32_t batchSize, int32_t branchDim,
    int32_t trunkDim, int32_t latentDim,
    int32_t querySize);

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    const int32_t batchSize = 2, branchDim = 4, trunkDim = 2;
    const int32_t latentDim = 8, querySize = 3;

    int32_t branchWSize = branchDim * latentDim + latentDim;
    int32_t trunkWSize = trunkDim * latentDim + latentDim;

    std::vector<float> hBranchIn(batchSize * branchDim, 1.0f);
    std::vector<float> hTrunkIn(querySize * trunkDim, 0.5f);
    std::vector<float> hBranchW(branchWSize, 0.1f);
    std::vector<float> hTrunkW(trunkWSize, 0.1f);
    std::vector<float> hOutput(batchSize * querySize, 0.0f);

    uint64_t wsSize = aclnnDeepOnetForwardGetWorkspaceSize(
        batchSize, branchDim, trunkDim, latentDim, querySize);

    void *dBI, *dTI, *dBW, *dTW, *dOut, *dWS;
    aclrtMalloc(&dBI, hBranchIn.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dTI, hTrunkIn.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dBW, hBranchW.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dTW, hTrunkW.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dOut, hOutput.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dWS, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpyAsync(dBI, hBranchIn.size() * sizeof(float), hBranchIn.data(),
                      hBranchIn.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(dTI, hTrunkIn.size() * sizeof(float), hTrunkIn.data(),
                      hTrunkIn.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(dBW, hBranchW.size() * sizeof(float), hBranchW.data(),
                      hBranchW.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtMemcpyAsync(dTW, hTrunkW.size() * sizeof(float), hTrunkW.data(),
                      hTrunkW.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);

    aclnnDeepOnetForward(dBI, dTI, dBW, dTW, dOut,
                          batchSize, branchDim, trunkDim, latentDim, querySize,
                          dWS, wsSize, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hOutput.data(), hOutput.size() * sizeof(float),
                 dOut, hOutput.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "DeepONet demo: output[0] = " << hOutput[0] << std::endl;
    std::cout << "DONE" << std::endl;

    aclrtFree(dBI); aclrtFree(dTI); aclrtFree(dBW);
    aclrtFree(dTW); aclrtFree(dOut); aclrtFree(dWS);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
