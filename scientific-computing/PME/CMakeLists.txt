# pme/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(ewald_reciprocal_op LANGUAGES CXX C)

# Find CANN
if(NOT DEFINED CANN_INC_DIR)
    set(CANN_INC_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include")
endif()
if(NOT DEFINED ASCENDCL_LIB)
    set(ASCENDCL_LIB "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64/libascendcl.so")
endif()

# ============================================================
# PME Kernel — Ascend C kernel (build with ascendc.cmake)
# ============================================================
option(ENABLE_ASCEND_KERNEL "Build Ascend C kernel" ON)

if(ENABLE_ASCEND_KERNEL)
    include(cmake/ascend.cmake)

    # Define the Ascend C kernel
    ascendc_library(pme_kernel SHARED
        op_kernel/pme_kernel.cpp
    )

    set_target_properties(pme_kernel PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )

    # Copy ascendc.cmake to cmake/ for independent builds
    configure_file(
        cmake/ascend.cmake
        ${CMAKE_BINARY_DIR}/cmake/ascend.cmake
        COPYONLY
    )
endif()

# ============================================================
# PME Host library — links the kernel .so
# ============================================================
add_library(pme_host2 STATIC
    op_host/pme_host2.cpp
)

target_include_directories(pme_host2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CANN_INC_DIR}
)

target_link_libraries(pme_host2 PUBLIC
    ${ASCENDCL_LIB}
)

# ============================================================
# Ewald Host library (original, kept for reference)
# ============================================================
add_library(ewald_host STATIC
    op_host/ewald_host.cpp
)

target_include_directories(ewald_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CANN_INC_DIR}
)

target_link_libraries(ewald_host PUBLIC
    ${ASCENDCL_LIB}
)

install(TARGETS ewald_host pme_host2 LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
