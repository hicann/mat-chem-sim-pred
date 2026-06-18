/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include <cstdlib>
#include "pid_basis_gemm_acl_smoke.h"
#include "pid_ipdt_basis_gemm_fit_host.h"

int main(int argc, char** argv)
{
    const int device_id = argc > 1 ? std::atoi(argv[1]) : 0;
    aclInit(nullptr);
    aclrtSetDevice(device_id);
    const int ret = pid_basis_gemm::RunAclSmoke(
        "PidIpdtBasisGemmFitUT", aclnnPidIpdtBasisGemmFitGetWorkspaceSize, aclnnPidIpdtBasisGemmFit);
    aclrtResetDevice(device_id);
    aclFinalize();
    return ret;
}
