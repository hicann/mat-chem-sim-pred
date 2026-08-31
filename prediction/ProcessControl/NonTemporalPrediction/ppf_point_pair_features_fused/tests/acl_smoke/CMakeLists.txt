# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

add_executable(test_ppf_point_pair_features_fused
    test_ppf_point_pair_features_fused.cpp)
target_include_directories(test_ppf_point_pair_features_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../../op_host)
target_link_directories(test_ppf_point_pair_features_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(test_ppf_point_pair_features_fused PRIVATE
    ppf_point_pair_features_fused_host ascendcl)
set_target_properties(test_ppf_point_pair_features_fused PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR};${CMAKE_BINARY_DIR}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64")
add_test(NAME ppf_point_pair_features_smoke
    COMMAND test_ppf_point_pair_features_fused 0)
set_tests_properties(ppf_point_pair_features_smoke PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64")
