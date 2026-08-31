/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "../../spectral_basis_kernel_common.h"

extern "C" __global__ __aicore__ void csr_tagcn_basis_k3_fused_stage1_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                             GM_ADDR norm, GM_ADDR features,
                                                                             GM_ADDR basis, GM_ADDR workspace,
                                                                             GM_ADDR tiling)
{
    (void)workspace;
    SpectralBasis::Run(rowPtr, sourceIndex, norm, features, basis, tiling, 4U, 0U);
}

extern "C" __global__ __aicore__ void csr_tagcn_basis_k3_fused_stage2_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                             GM_ADDR norm, GM_ADDR features,
                                                                             GM_ADDR basis, GM_ADDR workspace,
                                                                             GM_ADDR tiling)
{
    (void)workspace;
    SpectralBasis::Run(rowPtr, sourceIndex, norm, features, basis, tiling, 4U, 2U);
}

extern "C" __global__ __aicore__ void csr_tagcn_basis_k3_fused_stage3_kernel(GM_ADDR rowPtr, GM_ADDR sourceIndex,
                                                                             GM_ADDR norm, GM_ADDR features,
                                                                             GM_ADDR basis, GM_ADDR workspace,
                                                                             GM_ADDR tiling)
{
    (void)workspace;
    SpectralBasis::Run(rowPtr, sourceIndex, norm, features, basis, tiling, 4U, 3U);
}
