# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
add_executable(test_aclnn_csr_signed_cross_mean_pack_fused
    test_aclnn_csr_signed_cross_mean_pack_fused.cpp)
target_include_directories(test_aclnn_csr_signed_cross_mean_pack_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../op_host)
target_link_directories(test_aclnn_csr_signed_cross_mean_pack_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(test_aclnn_csr_signed_cross_mean_pack_fused PRIVATE
    csr_signed_cross_mean_pack_fused_host ascendcl)
add_test(NAME csr_signed_cross_mean_pack_smoke
    COMMAND test_aclnn_csr_signed_cross_mean_pack_fused 0)
set_tests_properties(csr_signed_cross_mean_pack_smoke PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}")
