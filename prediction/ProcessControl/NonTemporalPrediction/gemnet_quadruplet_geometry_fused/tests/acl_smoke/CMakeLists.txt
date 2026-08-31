# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

add_executable(test_gemnet_quadruplet_geometry_fused
    test_gemnet_quadruplet_geometry_fused.cpp)
target_include_directories(test_gemnet_quadruplet_geometry_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../../op_host)
target_link_directories(test_gemnet_quadruplet_geometry_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(test_gemnet_quadruplet_geometry_fused PRIVATE
    gemnet_quadruplet_geometry_fused_host ascendcl)
set_target_properties(test_gemnet_quadruplet_geometry_fused PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR};${CMAKE_BINARY_DIR}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64")
add_test(NAME gemnet_quadruplet_geometry_smoke
    COMMAND test_gemnet_quadruplet_geometry_fused 0)
set_tests_properties(gemnet_quadruplet_geometry_smoke PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64")
