# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

add_executable(test_aclnn_csr_gcn2_residual_propagate_fused
    test_aclnn_csr_gcn2_residual_propagate_fused.cpp)
target_include_directories(test_aclnn_csr_gcn2_residual_propagate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../op_host)
target_link_directories(test_aclnn_csr_gcn2_residual_propagate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
set(ASCEND_DRIVER_LIBS "")
if(EXISTS "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so")
    target_link_directories(test_aclnn_csr_gcn2_residual_propagate_fused PRIVATE
        /usr/local/Ascend/driver/lib64/driver)
    target_link_options(test_aclnn_csr_gcn2_residual_propagate_fused PRIVATE
        "-Wl,-rpath-link,/usr/local/Ascend/driver/lib64/driver")
    list(APPEND ASCEND_DRIVER_LIBS ascend_hal)
endif()
target_link_libraries(test_aclnn_csr_gcn2_residual_propagate_fused PRIVATE
    csr_gcn2_residual_propagate_fused_host ascendcl ${ASCEND_DRIVER_LIBS})
set_target_properties(test_aclnn_csr_gcn2_residual_propagate_fused PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR};${CMAKE_BINARY_DIR}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_options(test_aclnn_csr_gcn2_residual_propagate_fused PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64")
add_test(NAME csr_gcn2_residual_propagate_smoke
    COMMAND test_aclnn_csr_gcn2_residual_propagate_fused 0)
