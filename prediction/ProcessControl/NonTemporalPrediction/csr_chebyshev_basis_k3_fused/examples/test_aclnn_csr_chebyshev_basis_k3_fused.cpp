/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "../../spectral_acl_smoke_helpers.h"
#include "csr_chebyshev_basis_k3_fused_host.h"

int main(int argc, char** argv)
{
    const SpectralSmoke::BasisApi api{aclnnCsrChebyshevBasisK3FusedGetWorkspaceSize, aclnnCsrChebyshevBasisK3Fused, 3U};
    return SpectralSmoke::RunBasis(argc, argv, api);
}
