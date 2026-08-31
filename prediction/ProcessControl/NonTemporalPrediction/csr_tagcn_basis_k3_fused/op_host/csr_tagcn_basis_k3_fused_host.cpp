/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "csr_tagcn_basis_k3_fused_host.h"

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"

namespace
{
struct CsrTagcnBasisK3FusedTiling
{
    uint32_t nodes;
    uint32_t edges;
    uint32_t channels;
    uint32_t core_num;
};

uint64_t AlignUp(uint64_t value) { return (value + 31U) / 32U * 32U; }
}  // namespace

extern "C" uint32_t aclrtlaunch_csr_tagcn_basis_k3_fused_stage1_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                       void*, void*, void*, void*);
extern "C" uint32_t aclrtlaunch_csr_tagcn_basis_k3_fused_stage2_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                       void*, void*, void*, void*);
extern "C" uint32_t aclrtlaunch_csr_tagcn_basis_k3_fused_stage3_kernel(uint32_t, aclrtStream, void*, void*, void*,
                                                                       void*, void*, void*, void*);

extern "C" uint64_t aclnnCsrTagcnBasisK3FusedGetWorkspaceSize(int64_t, int64_t, int64_t)
{
    return AlignUp(sizeof(CsrTagcnBasisK3FusedTiling));
}

extern "C" int32_t aclnnCsrTagcnBasisK3Fused(void* rowPtr, void* sourceIndex, void* norm, void* features, void* basis,
                                             int64_t nodes, int64_t edges, int64_t channels, void* workspace,
                                             uint64_t workspaceSize, void* stream)
{
    const uint64_t required = aclnnCsrTagcnBasisK3FusedGetWorkspaceSize(nodes, edges, channels);
    if (rowPtr == nullptr || sourceIndex == nullptr || norm == nullptr || features == nullptr || basis == nullptr ||
        workspace == nullptr || stream == nullptr || workspaceSize < required || nodes <= 0 || nodes > INT32_MAX ||
        edges <= 0 || edges > INT32_MAX || channels <= 0 || channels > 4096)
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
    CsrTagcnBasisK3FusedTiling tiling{static_cast<uint32_t>(nodes), static_cast<uint32_t>(edges),
                                      static_cast<uint32_t>(channels), cores};
    const int32_t copied = aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    if (copied != ACL_SUCCESS)
    {
        return copied;
    }
    uint32_t result = aclrtlaunch_csr_tagcn_basis_k3_fused_stage1_kernel(
        cores, reinterpret_cast<aclrtStream>(stream), rowPtr, sourceIndex, norm, features, basis, workspace, workspace);
    if (result == 0U)
    {
        result = aclrtlaunch_csr_tagcn_basis_k3_fused_stage2_kernel(cores, reinterpret_cast<aclrtStream>(stream),
                                                                    rowPtr, sourceIndex, norm, features, basis,
                                                                    workspace, workspace);
    }
    if (result == 0U)
    {
        result = aclrtlaunch_csr_tagcn_basis_k3_fused_stage3_kernel(cores, reinterpret_cast<aclrtStream>(stream),
                                                                    rowPtr, sourceIndex, norm, features, basis,
                                                                    workspace, workspace);
    }
    return static_cast<int32_t>(result);
}
