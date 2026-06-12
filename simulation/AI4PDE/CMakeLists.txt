# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(ai4pde LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    else()
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_TOOLKIT_HOME})
    endif()
else()
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})
endif()

message(STATUS "Building AI4PDE operators")
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

set(SOC_VERSION "Ascend910B1" CACHE STRING "SOC version")
message(STATUS "SOC_VERSION: ${SOC_VERSION}")

set(RUN_MODE "npu" CACHE STRING "Run mode: npu or sim")
set(BUILD_MODE "normal" CACHE STRING "Build mode")
set(DYNAMIC_MODE OFF CACHE BOOL "Enable dynamic mode")
set(BUILD_TARGET "all" CACHE STRING "Build target: all, pinn, fno, deeponet, mesh_graph_net")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ASCEND_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake")
elseif(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/../8.0.RC1.alpha001/aarch64-linux/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/../8.0.RC1.alpha001/aarch64-linux/tikcpp/ascendc_kernel_cmake")
else()
    file(GLOB_RECURSE ASCENDC_CMAKE_FILES "${ASCEND_CANN_PACKAGE_PATH}/../*/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
    if(ASCENDC_CMAKE_FILES)
        list(GET ASCENDC_CMAKE_FILES 0 FIRST_CMAKE)
        get_filename_component(ASCENDC_CMAKE_PATH "${FIRST_CMAKE}" DIRECTORY)
    else()
        message(FATAL_ERROR "Cannot find ascendc_kernel_cmake directory.")
    endif()
endif()

message(STATUS "ASCENDC_CMAKE_PATH: ${ASCENDC_CMAKE_PATH}")
include(${ASCENDC_CMAKE_PATH}/ascendc.cmake)

if(BUILD_TARGET STREQUAL "all" OR BUILD_TARGET STREQUAL "pinn")
    add_subdirectory(pinn)
endif()
if(BUILD_TARGET STREQUAL "all" OR BUILD_TARGET STREQUAL "fno")
    add_subdirectory(fno)
endif()
if(BUILD_TARGET STREQUAL "all" OR BUILD_TARGET STREQUAL "deeponet")
    add_subdirectory(deeponet)
endif()
if(BUILD_TARGET STREQUAL "all" OR BUILD_TARGET STREQUAL "mesh_graph_net")
    add_subdirectory(mesh_graph_net)
endif()
