# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

add_executable(test_dimenet_triplet_angle_fused
    test_dimenet_triplet_angle_fused.cpp)
target_include_directories(test_dimenet_triplet_angle_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../../op_host)
target_link_directories(test_dimenet_triplet_angle_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(test_dimenet_triplet_angle_fused PRIVATE
    dimenet_triplet_angle_fused_host ascendcl)
set_target_properties(test_dimenet_triplet_angle_fused PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR};${CMAKE_BINARY_DIR}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64")
add_test(NAME dimenet_triplet_angle_smoke
    COMMAND test_dimenet_triplet_angle_fused 0)
set_tests_properties(dimenet_triplet_angle_smoke PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64")
