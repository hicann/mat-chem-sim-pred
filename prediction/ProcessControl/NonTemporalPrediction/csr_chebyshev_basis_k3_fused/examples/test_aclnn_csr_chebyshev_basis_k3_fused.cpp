/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "../../acl_smoke_common.h"
#include "csr_chebyshev_basis_k3_fused_host.h"

int main(int argc, char** argv)
{
    const AclSmoke::BasisApi api{aclnnCsrChebyshevBasisK3FusedGetWorkspaceSize, aclnnCsrChebyshevBasisK3Fused, 3U};
    return AclSmoke::RunBasis(argc, argv, api);
}
