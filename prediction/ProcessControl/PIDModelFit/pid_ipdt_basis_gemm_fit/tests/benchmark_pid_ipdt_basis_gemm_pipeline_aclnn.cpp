/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define PID_BASIS_PIPELINE_HOST_HEADER "pid_ipdt_basis_gemm_fit_host.h"
#define PID_BASIS_PIPELINE_GET_WORKSPACE aclnnPidIpdtBasisGemmFitGetWorkspaceSize
#define PID_BASIS_PIPELINE_RUN aclnnPidIpdtBasisGemmFit
#define PID_BASIS_PIPELINE_OP_NAME "PidIpdtBasisGemmPipeline"

#include "../../pid_fopdt_basis_gemm_fit/tests/benchmark_pid_fopdt_basis_gemm_pipeline_aclnn.cpp"
