# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
add_executable(test_aclnn_csr_genconv_softmax_aggregate_fused
    test_aclnn_csr_genconv_softmax_aggregate_fused.cpp)
target_include_directories(test_aclnn_csr_genconv_softmax_aggregate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../op_host)
target_link_directories(test_aclnn_csr_genconv_softmax_aggregate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
set(ASCEND_DRIVER_LIBS "")
if(EXISTS "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so")
    target_link_directories(test_aclnn_csr_genconv_softmax_aggregate_fused PRIVATE
        /usr/local/Ascend/driver/lib64/driver)
    target_link_options(test_aclnn_csr_genconv_softmax_aggregate_fused PRIVATE
        "-Wl,-rpath-link,/usr/local/Ascend/driver/lib64/driver")
    list(APPEND ASCEND_DRIVER_LIBS ascend_hal)
endif()
target_link_libraries(test_aclnn_csr_genconv_softmax_aggregate_fused PRIVATE
    csr_genconv_softmax_aggregate_fused_host ascendcl ${ASCEND_DRIVER_LIBS})
add_test(NAME csr_genconv_softmax_aggregate_smoke
    COMMAND test_aclnn_csr_genconv_softmax_aggregate_fused 0)
set(_ld_library_path
    "${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64:$ENV{LD_LIBRARY_PATH}")
set_tests_properties(csr_genconv_softmax_aggregate_smoke PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${_ld_library_path}")
