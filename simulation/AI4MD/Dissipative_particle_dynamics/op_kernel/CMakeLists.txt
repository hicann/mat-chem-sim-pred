# op_kernel/CMakeLists.txt
# DPD算子内核端构建配置

# 设置内核编译选项
set(KERNEL_SOURCES
    dpd_kernel.cpp
    dpd_tiling.cpp
)

set(KERNEL_HEADERS
    dpd_kernel.h
    dpd_tiling.h
)

# 设置CANN路径
set(CANN_INCLUDE_DIR ${CANN_PATH}/include)
set(CANN_LIB_DIR ${CANN_PATH}/lib64)

# 添加包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CANN_INCLUDE_DIR}
    ${CANN_INCLUDE_DIR}/acllib
    ${CANN_INCLUDE_DIR}/runtime
)

# 设置编译器标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=tsv110 -march=armv8-a")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__CCE_KT_TEST__ -D__CCE_KT_TEST__")

# 创建内核库
add_library(dpd_kernel SHARED ${KERNEL_SOURCES})

# 设置目标属性
set_target_properties(dpd_kernel PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# 链接库
target_link_libraries(dpd_kernel
    -Wl,--whole-archive
    ${CANN_LIB_DIR}/libcce.a
    -Wl,--no-whole-archive
    ${CANN_LIB_DIR}/libruntime.so
    ${CANN_LIB_DIR}/libascendcl.so
)

# 安装目标
install(TARGETS dpd_kernel
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${KERNEL_HEADERS}
    DESTINATION include/dpd
)

message(STATUS "内核端构建配置完成")
message(STATUS "内核源文件: ${KERNEL_SOURCES}")
message(STATUS "CANN包含目录: ${CANN_INCLUDE_DIR}")