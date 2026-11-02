# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
add_executable(test_csr_agnn_cosine_attention_aggregate_fused
    test_csr_agnn_cosine_attention_aggregate_fused.cpp)
target_include_directories(test_csr_agnn_cosine_attention_aggregate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../../op_host)
target_link_directories(test_csr_agnn_cosine_attention_aggregate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(test_csr_agnn_cosine_attention_aggregate_fused PRIVATE
    csr_agnn_cosine_attention_aggregate_fused_host ascendcl)
add_test(NAME csr_agnn_cosine_attention_aggregate_smoke
    COMMAND test_csr_agnn_cosine_attention_aggregate_fused 0)
