# npu_ops/gaff2/cmake/ascend.cmake
# Ascend C kernel build utilities
# Uses ccec (CANN C/C++ compiler) directly

# Check if already included
if(ASCEND_CMAKE_INCLUDED)
    return()
endif()
set(ASCEND_CMAKE_INCLUDED TRUE)

message(STATUS "[ascend.cmake] Loading...")

# Locate CANN base
set(CANN_BASE "${ASCEND_CANN_PACKAGE_PATH}")
if(NOT CANN_BASE OR NOT IS_DIRECTORY "${CANN_BASE}")
    if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
        set(CANN_BASE "$ENV{ASCEND_CANN_PACKAGE_PATH}")
    else()
        set(CANN_BASE "/mnt/Ascend/ascend-toolkit/8.1.RC1.alpha001")
    endif()
endif()
message(STATUS "[ascend.cmake] CANN_BASE=${CANN_BASE}")

# Locate ccec compiler
set(CCEC_BIN_DIR "${CANN_BASE}/aarch64-linux/ccec_compiler/bin")
find_program(ASCENDC_COMPILER ascendlmc
    PATHS ${CCEC_BIN_DIR}
    NO_DEFAULT_PATH
)
# Actually on this CANN version, the compiler is just ccec
find_program(CCEC_PATH ccec
    PATHS ${CCEC_BIN_DIR}
    NO_DEFAULT_PATH
)

if(NOT CCEC_PATH)
    message(WARNING "[ascend.cmake] ccec compiler not found at ${CCEC_BIN_DIR}")
else()
    message(STATUS "[ascend.cmake] ccec: ${CCEC_PATH}")
    # Add to PATH for subsequent builds
    set(ENV{PATH} "${CCEC_BIN_DIR}:$ENV{PATH}")
endif()

# Macro to build an Ascend C kernel
# Usage: build_ascendc_kernel(TARGET output.o SOURCES file1.h file2.cpp KERNEL_TYPE 1)
function(build_ascendc_kernel)
    set(options "")
    set(oneValueArgs TARGET KERNEL_TYPE SOC SOURCE_FILE)
    set(multiValueArgs)
    cmake_parse_arguments(AK "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT AK_SOC)
        set(AK_SOC "${SOC_VERSION}")
    endif()
    if(NOT AK_KERNEL_TYPE)
        set(AK_KERNEL_TYPE 1)
    endif()
    if(NOT AK_SOURCE_FILE)
        set(AK_SOURCE_FILE "op_kernel/${AK_TARGET}")
    endif()

    message(STATUS "[ascend.cmake] Building kernel: target=${AK_TARGET} soc=${AK_SOC}")

    # Compile kernel with ccec
    # Note: ccec is the CANN kernel compiler (ascendc is not a separate command)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${AK_TARGET}
        COMMAND ${CCEC_PATH}
            --md=${AK_SOC}
            --kernel-type=${AK_KERNEL_TYPE}
            -c ${AK_SOURCES}
            -o ${CMAKE_CURRENT_BINARY_DIR}/${AK_TARGET}
        DEPENDS ${AK_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Ascend C kernel: ${AK_TARGET}"
        VERBATIM
    )

    add_custom_target(${AK_TARGET}_kernel ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${AK_TARGET}
    )

    set(${AK_TARGET}_BIN "${CMAKE_CURRENT_BINARY_DIR}/${AK_TARGET}" PARENT_SCOPE)
    message(STATUS "[ascend.cmake] build_ascendc_kernel target created: ${AK_TARGET}")
endfunction()
