# shake/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(SHAKE LANGUAGES CXX)

# ============================================================
# CANN独立编译引导
# ============================================================
if(NOT ASCENDC_CMAKE_LOADED)
    if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "CANN path")
    endif()
    set(CANN_INC_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include")
    set(CANN_LIB_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64")
    find_library(ASCENDCL_LIB ascendcl PATHS ${CANN_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
    set(CANN_ASCENDC_CMAKE "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
    if(EXISTS "${CANN_ASCENDC_CMAKE}")
        include(${CANN_ASCENDC_CMAKE})
        set(ASCENDC_CMAKE_LOADED TRUE)
        message(STATUS "Standalone CANN: ${ASCEND_CANN_PACKAGE_PATH}")
    endif()
endif()
option(ENABLE_ASCEND_KERNEL "Build Ascend C kernel" ON)

# ============================================================
# Ascend C SHAKE iteration kernel
# ============================================================
if(ENABLE_ASCEND_KERNEL AND ASCENDC_CMAKE_LOADED)
    ascendc_library(shake_iteration_kernel SHARED
        op_kernel/shake.cpp
    )
    target_include_directories(shake_iteration_kernel_interface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    )
    get_target_property(_CUR_INCLUDES shake_iteration_kernel_interface INTERFACE_INCLUDE_DIRECTORIES)
    if(_CUR_INCLUDES)
        set_target_properties(shake_iteration_kernel_interface PROPERTIES INCLUDES "${_CUR_INCLUDES}")
    endif()

    # Extract kernel binary from merge object and convert to .inc
    # kernel binary = the entire ELF device_aiv.o file
    set(SHAKE_MERGE_OBJ
        ${CMAKE_CURRENT_BINARY_DIR}/shake_iteration_kernel_merge_obj_dir/device_aiv.o)
    set(SHAKE_KERNEL_INC
        ${CMAKE_CURRENT_BINARY_DIR}/shake_iteration_device_aiv_bin.inc)

    add_custom_command(OUTPUT ${SHAKE_KERNEL_INC}
        COMMAND xxd -i ${SHAKE_MERGE_OBJ} > ${SHAKE_KERNEL_INC}
        DEPENDS ${SHAKE_MERGE_OBJ}
        COMMENT "Converting SHAKE kernel binary to .inc (xxd -i device_aiv.o)"
    )

    add_custom_target(shake_kernel_inc_target DEPENDS ${SHAKE_KERNEL_INC})

    message(STATUS "SHAKE iteration kernel configured with CANN ascendc_library")
endif()

# ============================================================
# SHAKE NPU host library
# Links against the shake_iteration_kernel binary via .inc file
# ============================================================
add_library(shake_host STATIC
    op_host/shake_host.cpp
)

target_include_directories(shake_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_BINARY_DIR}      # For .inc files
    ${CANN_INC_DIR}
)

target_link_libraries(shake_host PUBLIC
    ${ASCENDCL_LIB}
    dl
)

add_dependencies(shake_host shake_kernel_inc_target)

# ============================================================
# Header-only library — CPU reference SHAKE (for backwards compatibility)
# ============================================================
add_library(shake_constraint INTERFACE)
target_include_directories(shake_constraint INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)
