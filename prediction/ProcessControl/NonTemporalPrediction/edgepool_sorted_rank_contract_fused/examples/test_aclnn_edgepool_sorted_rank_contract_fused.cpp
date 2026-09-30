/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cstdint>
#include <iostream>

#include "acl/acl.h"
#include "edgepool_sorted_rank_contract_fused_host.h"

namespace
{
void FreeBuffers(void* (&buffers)[8])
{
    for (void* buffer : buffers)
    {
        if (buffer != nullptr)
        {
            aclrtFree(buffer);
        }
    }
}
bool AllocateBuffers(void* (&buffers)[8], const size_t (&sizes)[8])
{
    for (size_t index = 0; index < 8; ++index)
    {
        if (aclrtMalloc(&buffers[index], sizes[index], ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        {
            FreeBuffers(buffers);
            return false;
        }
    }
    return true;
}
}  // namespace

int RunSmoke(aclrtStream stream)
{
    constexpr int64_t nodes = 4;
    constexpr int64_t edges = 3;
    int32_t order[edges] = {2, 0, 1};
    int32_t edge_index[edges * 2] = {0, 1, 2, 1, 2, 3};
    float score[edges] = {0.2F, 0.3F, 0.4F};
    int32_t cluster[nodes] = {};
    int32_t selected_edge[nodes] = {};
    float cluster_score[nodes] = {};
    int32_t counts[2] = {};
    void* buffers[8] = {};
    const size_t sizes[] = {sizeof(order),         sizeof(edge_index),    sizeof(score),  sizeof(cluster),
                            sizeof(selected_edge), sizeof(cluster_score), sizeof(counts), 1};
    if (!AllocateBuffers(buffers, sizes))
    {
        return 3;
    }
    auto& [order_device, edge_device, score_device, cluster_device, selected_device, cluster_score_device,
           counts_device, workspace_device] = buffers;
    aclrtMemcpy(order_device, sizeof(order), order, sizeof(order), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(edge_device, sizeof(edge_index), edge_index, sizeof(edge_index), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(score_device, sizeof(score), score, sizeof(score), ACL_MEMCPY_HOST_TO_DEVICE);
    const int32_t undersized = aclnnEdgepoolSortedRankContractFused(
        order_device, edge_device, score_device, cluster_device, selected_device, cluster_score_device, counts_device,
        nodes, edges, workspace_device, 0, stream);
    const int32_t result = aclnnEdgepoolSortedRankContractFused(order_device, edge_device, score_device, cluster_device,
                                                                selected_device, cluster_score_device, counts_device,
                                                                nodes, edges, workspace_device, 1, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(cluster, sizeof(cluster), cluster_device, sizeof(cluster), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(selected_edge, sizeof(selected_edge), selected_device, sizeof(selected_edge),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(cluster_score, sizeof(cluster_score), cluster_score_device, sizeof(cluster_score),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(counts, sizeof(counts), counts_device, sizeof(counts), ACL_MEMCPY_DEVICE_TO_HOST);
    const bool passed = undersized == ACL_ERROR_INVALID_PARAM && result == 0 && counts[0] == 2 && counts[1] == 2 &&
                        cluster[0] == 1 && cluster[1] == 1 && cluster[2] == 0 && cluster[3] == 0 &&
                        selected_edge[0] == 2 && selected_edge[1] == 0;
    std::cout << (passed ? "passed" : "failed") << " result=" << result << " selected=" << counts[0]
              << " clusters=" << counts[1] << std::endl;
    FreeBuffers(buffers);
    return passed ? 0 : 10;
}

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS)
    {
        return 1;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS)
    {
        return 2;
    }
    const int result = RunSmoke(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return result;
}
