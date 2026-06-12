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
project(deeponet LANGUAGES CXX)

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

set(SOC_VERSION "Ascend910B1" CACHE STRING "SOC version")
set(RUN_MODE "npu" CACHE STRING "Run mode: npu or sim")
set(BUILD_MODE "normal" CACHE STRING "Build mode")
set(DYNAMIC_MODE OFF CACHE BOOL "Enable dynamic mode")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ASCEND_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake")
else()
    file(GLOB_RECURSE ASCENDC_CMAKE_FILES "${ASCEND_CANN_PACKAGE_PATH}/../*/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
    if(ASCENDC_CMAKE_FILES)
        list(GET ASCENDC_CMAKE_FILES 0 FIRST_CMAKE)
        get_filename_component(ASCENDC_CMAKE_PATH "${FIRST_CMAKE}" DIRECTORY)
    else()
        message(FATAL_ERROR "Cannot find ascendc_kernel_cmake directory.")
    endif()
endif()

include(${ASCENDC_CMAKE_PATH}/ascendc.cmake)

set(KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/deeponet_kernel.cpp
)

ascendc_library(deeponet SHARED
    ${KERNEL_SOURCES}
)

ascendc_include_directories(deeponet PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../pde_common
)

set(HOST_SOURCES
    op_host/deeponet_host.cpp
)

add_library(deeponet_host SHARED ${HOST_SOURCES})

target_include_directories(deeponet_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_directories(deeponet_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(deeponet_host
    ascendcl
    deeponet
)

install(TARGETS deeponet_host LIBRARY DESTINATION lib)
install(FILES op_kernel/deeponet_kernel.h DESTINATION include)
