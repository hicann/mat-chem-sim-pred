# op_host/CMakeLists.txt
# DPD算子主机端构建配置

# 设置主机端源文件
set(HOST_SOURCES
    dpd_host.cpp
    dpd_host_continued.cpp
)

set(HOST_HEADERS
    dpd_host.h
    dpd_params.h
)

# 设置包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/op_kernel
    ${CANN_INCLUDE_DIR}
    ${CANN_INCLUDE_DIR}/acllib
    ${CANN_INCLUDE_DIR}/runtime
)

# 设置编译器标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

# 创建主机端库
add_library(dpd_host SHARED ${HOST_SOURCES})

# 设置目标属性
set_target_properties(dpd_host PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# 链接库
target_link_libraries(dpd_host
    # dpd_kernel  # 暂时跳过内核库
    ${CANN_LIB_DIR}/libascendcl.so
    ${CANN_LIB_DIR}/libruntime.so
    ${CANN_LIB_DIR}/libacl_op.so
    ${CANN_LIB_DIR}/libacl_op_compiler.so
)

# 添加Python绑定目标（可选）
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Development)
    if(Python3_FOUND)
        # 创建Python模块
        add_library(dpd_python MODULE dpd_python_bindings.cpp)
        target_link_libraries(dpd_python dpd_host pybind11::module)
        
        # 设置Python模块属性
        set_target_properties(dpd_python PROPERTIES
            PREFIX ""
            SUFFIX ".so"
        )
    endif()
endif()

# 安装目标
install(TARGETS dpd_host
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${HOST_HEADERS}
    DESTINATION include/dpd
)

# 创建可执行文件示例（可选）
# add_executable(dpd_demo dpd_demo.cpp)
# target_link_libraries(dpd_demo dpd_host)

message(STATUS "主机端构建配置完成")
message(STATUS "主机端源文件: ${HOST_SOURCES}")
message(STATUS "Python绑定: ${BUILD_PYTHON_BINDINGS}")