# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
cmake_minimum_required(VERSION 3.16)
project(dimenet_triplet_enumerate_fused LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_TOOLKIT_HOME})
endif()
if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()
set(SOC_VERSION "Ascend910B3" CACHE STRING "SOC version")
if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake")
elseif(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
else()
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake")
endif()
include(${ASCENDC_CMAKE_PATH}/ascendc.cmake)
ascendc_library(dimenet_triplet_enumerate_fused_kernel_lib SHARED
    op_kernel/dimenet_triplet_enumerate_fused_kernel.cpp)
add_library(dimenet_triplet_enumerate_fused_host SHARED
    op_host/dimenet_triplet_enumerate_fused_host.cpp)
target_include_directories(dimenet_triplet_enumerate_fused_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include op_host)
target_link_directories(dimenet_triplet_enumerate_fused_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(dimenet_triplet_enumerate_fused_host PRIVATE
    ascendcl dimenet_triplet_enumerate_fused_kernel_lib)
target_link_options(dimenet_triplet_enumerate_fused_host PRIVATE
    "-Wl,-rpath,$ORIGIN/lib")
add_library(dimenet_triplet_enumerate_fused_op_def SHARED
    op_host/dimenet_triplet_enumerate_fused_def.cpp)
target_include_directories(dimenet_triplet_enumerate_fused_op_def PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include)
enable_testing()
add_subdirectory(tests/acl_smoke)
install(TARGETS dimenet_triplet_enumerate_fused_host LIBRARY DESTINATION lib)
install(FILES op_host/dimenet_triplet_enumerate_fused_host.h DESTINATION include)
