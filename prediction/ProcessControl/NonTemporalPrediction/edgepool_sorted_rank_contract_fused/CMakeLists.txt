# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
cmake_minimum_required(VERSION 3.16)
project(edgepool_sorted_rank_contract_fused LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})
else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(SOC_VERSION "Ascend910B1" CACHE STRING "Ascend SoC version")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ASCEND_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake")
elseif(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
elseif(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake")
    set(ASCENDC_CMAKE_PATH "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake")
else()
    file(GLOB_RECURSE ASCENDC_CMAKE_FILES "${ASCEND_CANN_PACKAGE_PATH}/../*/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
    if(ASCENDC_CMAKE_FILES)
        list(GET ASCENDC_CMAKE_FILES 0 FIRST_CMAKE)
        get_filename_component(ASCENDC_CMAKE_PATH "${FIRST_CMAKE}" DIRECTORY)
    else()
        message(STATUS "Cannot find ascendc_kernel_cmake; skipping device kernel build on host without toolkit")
    endif()
endif()
if(ASCENDC_CMAKE_PATH)
    include(${ASCENDC_CMAKE_PATH}/ascendc.cmake)

    ascendc_library(edgepool_sorted_rank_contract_fused_kernel SHARED
        op_kernel/edgepool_sorted_rank_contract_fused_kernel.cpp)
    add_library(edgepool_sorted_rank_contract_fused_host SHARED
        op_host/edgepool_sorted_rank_contract_fused_host.cpp)
    target_include_directories(edgepool_sorted_rank_contract_fused_host PUBLIC
        ${ASCEND_CANN_PACKAGE_PATH}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
    target_link_directories(edgepool_sorted_rank_contract_fused_host PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64)
    target_link_libraries(edgepool_sorted_rank_contract_fused_host
        ascendcl edgepool_sorted_rank_contract_fused_kernel)
    add_library(edgepool_sorted_rank_contract_fused_op_def SHARED
        op_host/edgepool_sorted_rank_contract_fused_def.cpp)
    target_include_directories(edgepool_sorted_rank_contract_fused_op_def PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_host)

    if(BUILD_TESTING)
        enable_testing()
        add_subdirectory(examples)
    endif()

    install(TARGETS edgepool_sorted_rank_contract_fused_host
        edgepool_sorted_rank_contract_fused_op_def LIBRARY DESTINATION lib)
    install(FILES op_host/edgepool_sorted_rank_contract_fused_host.h
        DESTINATION include)
endif()
