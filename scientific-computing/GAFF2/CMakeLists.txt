# gaff2/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(gaff2_force_op LANGUAGES CXX C)

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
# Ascend C kernel build
# ascendc_library is loaded at root level — use it directly
if(ENABLE_ASCEND_KERNEL AND ASCENDC_CMAKE_LOADED)
    ascendc_library(gaff2_force_kernel SHARED
        op_kernel/gaff2_force.cpp
    )

    target_include_directories(gaff2_force_kernel_interface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    )

    # CANN's ascendc_library reads INCLUDES from ${target_name}_interface
    get_target_property(_CUR_INCLUDES gaff2_force_kernel_interface INTERFACE_INCLUDE_DIRECTORIES)
    if(_CUR_INCLUDES)
        set_target_properties(gaff2_force_kernel_interface PROPERTIES INCLUDES "${_CUR_INCLUDES}")
    endif()

    message(STATUS "gaff2_force_kernel configured with CANN ascendc_library")
elseif(ENABLE_ASCEND_KERNEL)
    # Fallback: try local ascend.cmake
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ascend.cmake")
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ascend.cmake)
    endif()
endif()

# Host library — provides the host-side API for launching this kernel
add_library(gaff2_force_host STATIC
    op_host/gaff2_host.cpp
    op_host/gaff2_def.cpp
)

target_include_directories(gaff2_force_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CANN_INC_DIR}
)

target_link_libraries(gaff2_force_host PUBLIC
    ${ASCENDCL_LIB}
    dl
)
