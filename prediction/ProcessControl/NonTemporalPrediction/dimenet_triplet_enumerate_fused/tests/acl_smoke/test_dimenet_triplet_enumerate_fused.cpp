/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../../acl_smoke_helpers.h"
#include "dimenet_triplet_enumerate_fused_host.h"

using MolecularGeometrySmoke::Check;
using MolecularGeometrySmoke::CopyIn;

bool CheckEnumeration(const std::vector<int32_t>& idxI, const std::vector<int32_t>& idxJ,
                      const std::vector<int32_t>& idxK, const std::vector<int32_t>& idxKj,
                      const std::vector<int32_t>& idxJi, const std::vector<int32_t>& counts)
{
    return counts[0] == 1 && counts[1] == 0 && idxI[0] == 2 && idxJ[0] == 1 && idxK[0] == 0 && idxKj[0] == 0 &&
           idxJi[0] == 2;
}

void CopyOutputs(std::vector<int32_t>& host, void* device)
{
    MolecularGeometrySmoke::Check(aclrtMemcpy(host.data(), host.size() * sizeof(int32_t), device,
                                              host.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
                                  "copy out");
}

int RunSmoke(int device)
{
    Check(aclInit(nullptr), "init");
    Check(aclrtSetDevice(device), "device");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "stream");
    const std::vector<int32_t> rowPtr{0, 0, 2, 3};
    const std::vector<int32_t> sourceIndex{0, 2, 1};
    constexpr int64_t capacity = 4;
    std::vector<int32_t> idxI(capacity, -1), idxJ(capacity, -1);
    std::vector<int32_t> idxK(capacity, -1), idxKj(capacity, -1);
    std::vector<int32_t> idxJi(capacity, -1), counts(2, -1);
    std::vector<void*> d(8, nullptr);
    void* workspace = nullptr;
    CopyIn(&d[0], rowPtr.data(), rowPtr.size() * sizeof(int32_t));
    CopyIn(&d[1], sourceIndex.data(), sourceIndex.size() * sizeof(int32_t));
    const size_t outputBytes = capacity * sizeof(int32_t);
    for (size_t index = 2; index < 7; ++index)
        Check(aclrtMalloc(&d[index], outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), "output");
    Check(aclrtMalloc(&d[7], 2 * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "counts");
    const uint64_t bytes = aclnnDimeNetTripletEnumerateFusedGetWorkspaceSize(3, 3, capacity);
    Check(aclrtMalloc(&workspace, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "workspace");
    const auto invoke = [&](uint64_t workspaceSize)
    {
        return aclnnDimeNetTripletEnumerateFused(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], 3, 3, capacity,
                                                 workspace, workspaceSize, stream);
    };
    if (invoke(0) != ACL_ERROR_INVALID_PARAM || invoke(bytes) != ACL_SUCCESS)
    {
        return 1;
    }
    Check(aclrtSynchronizeStream(stream), "sync");
    CopyOutputs(idxI, d[2]);
    CopyOutputs(idxJ, d[3]);
    CopyOutputs(idxK, d[4]);
    CopyOutputs(idxKj, d[5]);
    CopyOutputs(idxJi, d[6]);
    CopyOutputs(counts, d[7]);
    const bool passed = CheckEnumeration(idxI, idxJ, idxK, idxKj, idxJi, counts);
    aclrtFree(workspace);
    for (void* pointer : d) aclrtFree(pointer);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return passed ? 0 : 10;
}

int main(int argc, char** argv) { return RunSmoke(argc > 1 ? std::atoi(argv[1]) : 0); }
