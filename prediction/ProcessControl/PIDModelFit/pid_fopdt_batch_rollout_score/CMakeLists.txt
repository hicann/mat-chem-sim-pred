# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(pid_fopdt_batch_rollout_score LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    else()
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_TOOLKIT_HOME})
    endif()
else()
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})
endif()

set(SOC_VERSION "Ascend910B1" CACHE STRING "SOC version")
set(RUN_MODE "npu" CACHE STRING "Run mode: npu or sim")
set(BUILD_MODE "normal" CACHE STRING "Build mode")
set(DYNAMIC_MODE OFF CACHE BOOL "Enable dynamic mode")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ASCEND_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake")
elseif(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
elseif(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake")
else()
    file(GLOB_RECURSE ASCENDC_CMAKE_FILES
        "${ASCEND_CANN_PACKAGE_PATH}/../*/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
        "${ASCEND_CANN_PACKAGE_PATH}/../*/*-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    )
    if(ASCENDC_CMAKE_FILES)
        list(GET ASCENDC_CMAKE_FILES 0 FIRST_CMAKE)
        get_filename_component(ASCENDC_CMAKE_PATH "${FIRST_CMAKE}" DIRECTORY)
    else()
        message(FATAL_ERROR "Cannot find ascendc_kernel_cmake directory.")
    endif()
endif()

include(${ASCENDC_CMAKE_PATH}/ascendc.cmake)

ascendc_library(pid_fopdt_batch_rollout_score_kernel_lib SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/pid_fopdt_batch_rollout_score_kernel.cpp
)

ascendc_include_directories(pid_fopdt_batch_rollout_score_kernel_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

add_library(pid_fopdt_batch_rollout_score_host SHARED
    op_host/pid_fopdt_batch_rollout_score_host.cpp
)

target_include_directories(pid_fopdt_batch_rollout_score_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_link_directories(pid_fopdt_batch_rollout_score_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(pid_fopdt_batch_rollout_score_host
    ascendcl
    pid_fopdt_batch_rollout_score_kernel_lib
)

add_executable(test_aclnn_pid_fopdt_batch_rollout_score
    examples/test_aclnn_pid_fopdt_batch_rollout_score.cpp
)

target_include_directories(test_aclnn_pid_fopdt_batch_rollout_score PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_link_directories(test_aclnn_pid_fopdt_batch_rollout_score PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/lib
)

target_link_libraries(test_aclnn_pid_fopdt_batch_rollout_score
    ascendcl
    pid_fopdt_batch_rollout_score_host
)

add_executable(benchmark_pid_fopdt_batch_rollout_score
    tests/benchmark_pid_fopdt_batch_rollout_score_aclnn.cpp
)

target_include_directories(benchmark_pid_fopdt_batch_rollout_score PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_link_directories(benchmark_pid_fopdt_batch_rollout_score PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/lib
)

target_link_libraries(benchmark_pid_fopdt_batch_rollout_score
    ascendcl
    pid_fopdt_batch_rollout_score_host
)

install(TARGETS pid_fopdt_batch_rollout_score_host LIBRARY DESTINATION lib)
install(FILES op_host/pid_fopdt_batch_rollout_score_host.h DESTINATION include)
