# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
add_executable(test_aclnn_csr_transformer_dot_attention_aggregate_fused
    test_aclnn_csr_transformer_dot_attention_aggregate_fused.cpp)
target_include_directories(test_aclnn_csr_transformer_dot_attention_aggregate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../op_host)
target_link_directories(test_aclnn_csr_transformer_dot_attention_aggregate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(test_aclnn_csr_transformer_dot_attention_aggregate_fused PRIVATE
    csr_transformer_dot_attention_aggregate_fused_host ascendcl)
add_test(NAME csr_transformer_dot_attention_aggregate_smoke
    COMMAND test_aclnn_csr_transformer_dot_attention_aggregate_fused 0)
