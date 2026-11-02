# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
add_executable(test_aclnn_csr_sage_mean_root_relu_fused test_aclnn_csr_sage_mean_root_relu_fused.cpp)
target_include_directories(test_aclnn_csr_sage_mean_root_relu_fused PRIVATE
    ../op_host ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_directories(test_aclnn_csr_sage_mean_root_relu_fused PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(test_aclnn_csr_sage_mean_root_relu_fused PRIVATE csr_sage_mean_root_relu_fused_host ascendcl)
add_test(NAME csr_sage_mean_root_relu_smoke COMMAND test_aclnn_csr_sage_mean_root_relu_fused 0)
