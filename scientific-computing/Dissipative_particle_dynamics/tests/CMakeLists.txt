# tests/CMakeLists.txt
# DPD算子测试构建配置

# 设置测试选项
option(BUILD_TESTS "构建测试" ON)
option(BUILD_PYTHON_TESTS "构建Python测试" ON)

if(BUILD_TESTS)
    # 查找Google Test
    find_package(GTest REQUIRED)
    
    # 单元测试配置
    add_subdirectory(ut)
    
    # 安装测试数据
    install(DIRECTORY test_dpd_op.py
        DESTINATION tests
    )
    
    # 安装测试数据文件
    install(DIRECTORY ut/op_kernel/dpd_test_data/
        DESTINATION tests/data
    )
endif()

if(BUILD_PYTHON_TESTS AND Python3_FOUND)
    # 安装Python测试脚本
    install(FILES test_dpd_op.py
        DESTINATION tests
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
    )
    
    # 创建测试数据目录
    install(CODE "
        file(MAKE_DIRECTORY \${CMAKE_INSTALL_PREFIX}/tests/data)
    ")
endif()

message(STATUS "测试构建配置完成")
message(STATUS "单元测试: ${BUILD_TESTS}")
message(STATUS "Python测试: ${BUILD_PYTHON_TESTS}")