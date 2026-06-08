# velocity-verlet/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(integrator LANGUAGES CXX C)

# ============================================================
# CANN独立编译引导 (仅在不是从顶层加载时激活)
# ============================================================
if(NOT ASCENDC_CMAKE_LOADED)
    if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "CANN package path")
    endif()
    set(CANN_INC_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include")
    set(CANN_LIB_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64")
    find_library(ASCENDCL_LIB ascendcl PATHS ${CANN_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
    
    # Load CANN ascendc.cmake
    set(CANN_ASCENDC_CMAKE "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
    if(EXISTS "${CANN_ASCENDC_CMAKE}")
        include(${CANN_ASCENDC_CMAKE})
        set(ASCENDC_CMAKE_LOADED TRUE)
        message(STATUS "Standalone CANN bootstrap: ${ASCEND_CANN_PACKAGE_PATH}")
    else()
        message(WARNING "CANN ascendc.cmake not found at ${CANN_ASCENDC_CMAKE}")
    endif()
endif()

option(ENABLE_ASCEND_KERNEL "Build Ascend C kernel operators" ON)
# Ascend C kernel build (ascendc_library loaded at root level)
if(ENABLE_ASCEND_KERNEL AND ASCENDC_CMAKE_LOADED)
    # Build 3 kernels

    ascendc_library(vv_integrate_kernel SHARED
        op_kernel/vv_integrate.cpp
    )
    target_include_directories(vv_integrate_kernel_interface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    )
    get_target_property(_CUR_INCLUDES vv_integrate_kernel_interface INTERFACE_INCLUDE_DIRECTORIES)
    if(_CUR_INCLUDES)
        set_target_properties(vv_integrate_kernel_interface PROPERTIES INCLUDES "${_CUR_INCLUDES}")
    endif()

    ascendc_library(vv_finish_kernel SHARED
        op_kernel/vv_finish.cpp
    )
    target_include_directories(vv_finish_kernel_interface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    )
    get_target_property(_CUR_INCLUDES vv_finish_kernel_interface INTERFACE_INCLUDE_DIRECTORIES)
    if(_CUR_INCLUDES)
        set_target_properties(vv_finish_kernel_interface PROPERTIES INCLUDES "${_CUR_INCLUDES}")
    endif()

    ascendc_library(thermo_scale_kernel SHARED
        op_kernel/thermo_scale.cpp
    )
    target_include_directories(thermo_scale_kernel_interface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    )
    get_target_property(_CUR_INCLUDES thermo_scale_kernel_interface INTERFACE_INCLUDE_DIRECTORIES)
    if(_CUR_INCLUDES)
        set_target_properties(thermo_scale_kernel_interface PROPERTIES INCLUDES "${_CUR_INCLUDES}")
    endif()

    message(STATUS "VV integrator kernels configured with CANN ascendc_library")
endif()

# Host library — provides the host-side API for launching these kernels
add_library(vv_host STATIC
    op_host/vv_host.cpp
)

target_include_directories(vv_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_SOURCE_DIR}/operators/ensemble/op_host
    ${CANN_INC_DIR}
)

target_link_libraries(vv_host PUBLIC
    ${ASCENDCL_LIB}
    dl
)
