# examples/CMakeLists.txt
# DPD算子示例构建配置

# C++示例
add_executable(dpd_ascendc_demo
    dpd_ascendc_demo.cpp
)

target_link_libraries(dpd_ascendc_demo
    dpd_host
    dpd_kernel
)

# 设置可执行文件属性
set_target_properties(dpd_ascendc_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 安装示例
install(TARGETS dpd_ascendc_demo
    RUNTIME DESTINATION bin
)

install(FILES
    dpd_pytorch_demo.py
    DESTINATION examples
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
)

# 创建示例数据目录
install(CODE "
    file(MAKE_DIRECTORY \${CMAKE_INSTALL_PREFIX}/examples/data)
")

# 生成示例配置文件
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/example_config.json.in
    ${CMAKE_BINARY_DIR}/examples/example_config.json
    @ONLY
)

install(FILES
    ${CMAKE_BINARY_DIR}/examples/example_config.json
    DESTINATION examples
)

message(STATUS "示例构建配置完成")
message(STATUS "C++示例: dpd_ascendc_demo")
message(STATUS "Python示例: dpd_pytorch_demo.py")