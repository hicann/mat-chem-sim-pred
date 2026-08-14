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
project(reformer_lsh_bucket_sort LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_TOOLKIT_HOME})
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
else()
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})
endif()
set(SOC_VERSION "Ascend910B1" CACHE STRING "SOC version")

foreach(candidate
    "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
    if(EXISTS "${candidate}")
        set(ASCENDC_CMAKE_PATH "${candidate}")
        break()
    endif()
endforeach()
if(NOT ASCENDC_CMAKE_PATH)
    message(FATAL_ERROR "Cannot find ascendc_kernel_cmake")
endif()
include(${ASCENDC_CMAKE_PATH}/ascendc.cmake)

ascendc_library(reformer_lsh_bucket_sort_kernel_lib SHARED op_kernel/reformer_lsh_bucket_sort_kernel.cpp)
ascendc_include_directories(reformer_lsh_bucket_sort_kernel_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
add_library(reformer_lsh_bucket_sort_op_def SHARED
    op_host/reformer_lsh_bucket_sort_host.cpp
    op_host/reformer_lsh_bucket_sort_def.cpp)
target_include_directories(reformer_lsh_bucket_sort_op_def PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
install(FILES msopgen/reformer_lsh_bucket_sort_msopgen.json DESTINATION msopgen)
