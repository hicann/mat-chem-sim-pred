# tests/ut/CMakeLists.txt
# 单元测试构建配置

# 设置包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/op_kernel
    ${CMAKE_SOURCE_DIR}/op_host
    ${CANN_INCLUDE_DIR}
    ${CANN_INCLUDE_DIR}/acllib
    ${CANN_INCLUDE_DIR}/runtime
    ${GTEST_INCLUDE_DIRS}
)

# 内核单元测试
add_executable(dpd_kernel_test
    op_kernel/dpd_kernel_test.cpp
)

# 链接库
target_link_libraries(dpd_kernel_test
    dpd_kernel
    dpd_host
    GTest::gtest
    GTest::gtest_main
    ${CANN_LIB_DIR}/libascendcl.so
    ${CANN_LIB_DIR}/libruntime.so
)

# 添加测试
add_test(NAME dpd_kernel_test
    COMMAND dpd_kernel_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 设置测试属性
set_tests_properties(dpd_kernel_test PROPERTIES
    TIMEOUT 30
    LABELS "unit;kernel"
)

# 主机端单元测试（可选）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
    add_executable(dpd_host_test
        op_host/dpd_host_test.cpp
    )
    
    target_link_libraries(dpd_host_test
        dpd_host
        GTest::gtest
        GTest::gtest_main
    )
    
    add_test(NAME dpd_host_test
        COMMAND dpd_host_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    set_tests_properties(dpd_host_test PROPERTIES
        TIMEOUT 30
        LABELS "unit;host"
    )
endif()

# 性能测试（可选）
if(BUILD_PERF_TESTS)
    add_executable(dpd_perf_test
        perf/dpd_perf_test.cpp
    )
    
    target_link_libraries(dpd_perf_test
        dpd_host
        dpd_kernel
    )
    
    add_test(NAME dpd_perf_test
        COMMAND dpd_perf_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    set_tests_properties(dpd_perf_test PROPERTIES
        TIMEOUT 60
        LABELS "perf"
    )
endif()

message(STATUS "单元测试配置完成")