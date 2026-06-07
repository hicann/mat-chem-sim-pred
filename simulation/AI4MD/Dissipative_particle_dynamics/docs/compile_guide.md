# DPD算子编译指南

## 概述

本文档提供DPD（耗散粒子动力学）Ascend C算子的完整编译指南，包括环境配置、编译步骤、常见问题和部署说明。

## 1. 环境要求

### 1.1 硬件要求

#### 必需硬件
- **NPU设备**: 华为昇腾910B或昇腾310P AI处理器
- **CPU**: x86_64或ARMv8架构，≥ 8核心
- **内存**: ≥ 16GB RAM
- **存储**: ≥ 50GB可用空间

#### 推荐配置
- **NPU**: 昇腾910B（32个AI Core）
- **CPU**: 64核心，2.6GHz+
- **内存**: 64GB RAM
- **存储**: NVMe SSD，≥ 200GB

### 1.2 软件要求

#### 操作系统
- **Ubuntu**: 18.04/20.04/22.04 LTS
- **CentOS**: 7.6/7.9/8.2
- **EulerOS**: 2.0/2.8
- **KylinOS**: V10

#### 基础软件
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    libssl-dev

# CentOS/RHEL
sudo yum install -y \
    gcc-c++ \
    cmake \
    git \
    python3 \
    python3-pip \
    openssl-devel
```

#### CANN工具包
- **版本**: CANN 5.0.RC2 或更高版本
- **下载**: [华为昇腾社区](https://www.hiascend.com/software/cann)
- **安装路径**: 默认 `/usr/local/Ascend`

#### Python依赖（可选）
```bash
pip3 install torch numpy pandas matplotlib scikit-learn
```

## 2. 环境配置

### 2.1 设置环境变量

创建环境配置脚本 `setup_env.sh`:

```bash
#!/bin/bash
# DPD算子环境配置脚本

# CANN路径
export CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_TOOLKIT_HOME=$CANN_PATH

# 编译器路径
export CC=$CANN_PATH/bin/aarch64-linux-gcc
export CXX=$CANN_PATH/bin/aarch64-linux-g++

# 包含路径
export C_INCLUDE_PATH=$CANN_PATH/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CANN_PATH/include:$CPLUS_INCLUDE_PATH

# 库路径
export LD_LIBRARY_PATH=$CANN_PATH/lib64:$CANN_PATH/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CANN_PATH/lib64:$CANN_PATH/lib:$LIBRARY_PATH

# NPU设备设置
export ASCEND_OPP_PATH=$CANN_PATH/opp
export ASCEND_AICPU_PATH=$CANN_PATH
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=3

# 任务队列设置
export TASK_QUEUE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=600

echo "环境配置完成"
echo "CANN路径: $CANN_PATH"
echo "编译器: $CXX"
```

加载环境变量:
```bash
source setup_env.sh
```

### 2.2 验证环境

运行环境验证脚本 `check_env.py`:

```python
#!/usr/bin/env python3
import os
import subprocess
import sys

def check_environment():
    """检查编译环境"""
    
    checks = []
    
    # 检查CANN路径
    cann_path = os.environ.get('ASCEND_TOOLKIT_HOME')
    if cann_path and os.path.exists(cann_path):
        checks.append(('CANN路径', True, cann_path))
    else:
        checks.append(('CANN路径', False, '未找到'))
    
    # 检查编译器
    try:
        result = subprocess.run(['which', 'aarch64-linux-g++'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            checks.append(('编译器', True, result.stdout.strip()))
        else:
            checks.append(('编译器', False, '未找到'))
    except:
        checks.append(('编译器', False, '检查失败'))
    
    # 检查Python
    checks.append(('Python版本', True, sys.version))
    
    # 输出结果
    print("环境检查:")
    print("-" * 50)
    for name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}: {detail}")
    
    # 返回状态
    return all(passed for _, passed, _ in checks)

if __name__ == "__main__":
    if check_environment():
        print("\n✅ 环境检查通过")
        sys.exit(0)
    else:
        print("\n❌ 环境检查失败")
        sys.exit(1)
```

## 3. 编译步骤

### 3.1 获取源代码

```bash
# 克隆仓库
git clone https://github.com/your-org/dpd-operator.git
cd dpd-operator

# 或使用已下载的代码
cd /mnt/transing/dpd算子
```

### 3.2 配置编译选项

创建构建目录并配置CMake:

```bash
mkdir build && cd build

# 基本配置
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCANN_PATH=$ASCEND_TOOLKIT_HOME \
    -DBUILD_TESTS=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_EXAMPLES=ON

# 高级配置选项
# -DUSE_FP16=ON          # 启用fp16支持
# -DUSE_BF16=OFF         # 禁用bf16支持
# -DENABLE_DEBUG=OFF     # 禁用调试符号
# -DWITH_PROFILING=ON    # 启用性能分析
```

### 3.3 编译项目

```bash
# 并行编译（推荐）
make -j$(nproc)

# 或指定线程数
make -j16

# 详细输出
make VERBOSE=1
```

### 3.4 安装

```bash
# 安装到系统目录
sudo make install

# 或安装到自定义目录
make install DESTDIR=/opt/dpd-operator
```

### 3.5 验证编译

```bash
# 检查生成的文件
ls -la bin/ lib/ include/

# 运行单元测试
cd build
ctest --output-on-failure

# 运行Python测试
python3 ../tests/test_dpd_op.py
```

## 4. 快速编译脚本

创建一键编译脚本 `build.sh`:

```bash
#!/bin/bash
# DPD算子一键编译脚本

set -e  # 遇到错误立即退出

echo "🚀 开始编译DPD算子"
echo "="*50

# 1. 检查环境
echo "1. 检查环境..."
source setup_env.sh 2>/dev/null || {
    echo "❌ 环境配置失败"
    exit 1
}

# 2. 清理旧构建
echo "2. 清理旧构建..."
rm -rf build/ 2>/dev/null || true

# 3. 创建构建目录
echo "3. 创建构建目录..."
mkdir -p build && cd build

# 4. 配置CMake
echo "4. 配置CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCANN_PATH=$ASCEND_TOOLKIT_HOME \
    -DBUILD_TESTS=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_EXAMPLES=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local

if [ $? -ne 0 ]; then
    echo "❌ CMake配置失败"
    exit 1
fi

# 5. 编译
echo "5. 编译项目..."
CPU_COUNT=$(nproc)
make -j$CPU_COUNT

if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

# 6. 运行测试
echo "6. 运行测试..."
ctest --output-on-failure

if [ $? -ne 0 ]; then
    echo "⚠ 测试失败，继续安装..."
fi

# 7. 安装
echo "7. 安装..."
sudo make install

# 8. 验证安装
echo "8. 验证安装..."
if [ -f /usr/local/bin/dpd_ascendc_demo ]; then
    echo "✅ 编译安装成功!"
    echo ""
    echo "生成文件:"
    echo "  - 可执行文件: /usr/local/bin/dpd_ascendc_demo"
    echo "  - 库文件: /usr/local/lib/libdpd_*.so"
    echo "  - 头文件: /usr/local/include/dpd/"
    echo "  - 示例: /usr/local/examples/"
    echo ""
    echo "运行示例:"
    echo "  $ dpd_ascendc_demo"
    echo "  $ python3 /usr/local/examples/dpd_pytorch_demo.py"
else
    echo "❌ 安装验证失败"
    exit 1
fi

echo "="*50
echo "🎉 DPD算子编译完成!"
```

赋予执行权限并运行:
```bash
chmod +x build.sh
./build.sh
```

## 5. 交叉编译

### 5.1 为不同架构编译

```bash
# 为ARM架构编译
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-toolchain.cmake \
    -DCANN_PATH=/path/to/cann

# 为x86架构编译（模拟模式）
cmake .. \
    -DCMAKE_CXX_FLAGS="-D__CCE_KT_TEST__" \
    -DBUILD_FOR_SIMULATION=ON
```

### 5.2 工具链文件示例

创建 `cmake/aarch64-toolchain.cmake`:

```cmake
# 交叉编译工具链配置
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 编译器路径
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

# 搜索路径
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

# 搜索规则
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

## 6. 容器化编译

### 6.1 Dockerfile示例

```dockerfile
# 基于CANN官方镜像
FROM swr.cn-north-4.myhuaweicloud.com/ascend/cann:5.0.RC2.alpha005-ubuntu18.04-aarch64

# 设置工作目录
WORKDIR /workspace

# 安装依赖
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 复制源代码
COPY . .

# 设置环境变量
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64:$LD_LIBRARY_PATH

# 编译
RUN mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCANN_PATH=$ASCEND_TOOLKIT_HOME && \
    make -j$(nproc) && \
    make install

# 设置入口点
ENTRYPOINT ["/workspace/build/bin/dpd_ascendc_demo"]
```

### 6.2 构建和运行容器

```bash
# 构建镜像
docker build -t dpd-operator:latest .

# 运行容器
docker run --rm \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /var/log/npu/:/var/log/npu \
    dpd-operator:latest
```

## 7. 常见问题

### 7.1 编译错误

#### 问题1: 找不到CANN头文件
```
fatal error: acl/acl.h: No such file or directory
```

**解决方案**:
```bash
# 检查环境变量
echo $ASCEND_TOOLKIT_HOME

# 确保头文件存在
ls $ASCEND_TOOLKIT_HOME/include/acl/acl.h

# 重新配置CMake
rm -rf build && mkdir build && cd build
cmake .. -DCANN_PATH=$ASCEND_TOOLKIT_HOME
```

#### 问题2: 链接错误
```
undefined reference to `aclInit'
```

**解决方案**:
```bash
# 检查库路径
echo $LD_LIBRARY_PATH

# 确保库文件存在
ls $ASCEND_TOOLKIT_HOME/lib64/libascendcl.so

# 重新链接
make clean && make
```

#### 问题3: 内存不足
```
c++: fatal error: Killed signal terminated program cc1plus
```

**解决方案**:
```bash
# 减少并行编译数
make -j4

# 增加交换空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 7.2 运行时错误

#### 问题1: NPU设备不可用
```
ACL error: device not found
```

**解决方案**:
```bash
# 检查NPU设备
npu-smi info

# 检查设备权限
ls -la /dev/davinci*

# 添加用户到HwHiAiUser组
sudo usermod -a -G HwHiAiUser $USER
```

#### 问题2: 版本不兼容
```
version `CANN_5.0' not found
```

**解决方案**:
```bash
# 检查CANN版本
cat $ASCEND_TOOLKIT_HOME/version.info

# 重新编译匹配版本
git checkout tags/v1.0.0-cann5.0
```

### 7.3 性能问题

#### 问题1: 编译时间过长
**解决方案**:
```bash
# 使用ccache加速
sudo apt-get install ccache
export CC="ccache gcc"
export CXX="ccache g++"

# 禁用调试符号
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_DEBUG=OFF
```

#### 问题2: 内存使用过高
**解决方案**:
```bash
# 使用Gold链接器
export CXXFLAGS="-fuse-ld=gold"

# 启用LTO优化
cmake .. -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
```

## 8. 性能优化编译

### 8.1 优化级别

```bash
# O0: 无优化（调试）
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-O0 -g"

# O2: 平衡优化（默认）
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2"

# O3: 激进优化
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"

# Ofast: 最快速度（可能影响精度）
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Ofast"
```

### 8.2 特定优化

```bash
# 向量化优化
-DCMAKE_CXX_FLAGS="-march=armv8.2-a+fp16+dotprod+sve"

# 链接时优化
-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

# 精简二进制
-DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections"
-DCMAKE_EXE_LINKER_FLAGS="-Wl,--gc-sections"
```

## 9. 部署指南

### 9.1 系统部署

```bash
# 1. 打包
tar czf dpd-operator-1.0.0.tar.gz \
    bin/dpd_ascendc_demo \
    lib/libdpd_*.so \
    include/dpd/ \
    examples/ \
    docs/

# 2. 传输到目标系统
scp dpd-operator-1.0.0.tar.gz user@target:/opt/

# 3. 目标系统安装
tar xzf dpd-operator-1.0.0.tar.gz -C /usr/local/
ldconfig
```

### 9.2 环境检查脚本

创建部署验证脚本 `deploy_check.sh`:

```bash
#!/bin/bash
echo "部署验证检查"
echo "=============="

# 检查文件
echo "1. 检查文件..."
files=(
    "/usr/local/bin/dpd_ascendc_demo"
    "/usr/local/lib/libdpd_host.so"
    "/usr/local/lib/libdpd_kernel.so"
    "/usr/local/include/dpd/dpd_host.h"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (缺失)"
    fi
done

# 检查环境变量
echo -e "\n2. 检查环境变量..."
env_vars=(
    "ASCEND_TOOLKIT_HOME"
    "LD_LIBRARY_PATH"
    "CANN_PATH"
)

for var in "${env_vars[@]}"; do
    value="${!var}"
    if [ -n "$value" ]; then
        echo "✅ $var=$value"
    else
        echo "❌ $var (未设置)"
    fi
done

# 检查NPU设备
echo -e "\n3. 检查NPU设备..."
if command -v npu-smi &> /dev/null; then
    npu-smi info
else
    echo "⚠ npu-smi 未找到"
fi

# 运行简单测试
echo -e "\n4. 运行简单测试..."
if [ -f "/usr/local/bin/dpd_ascendc_demo" ]; then
    timeout 10 /usr/local/bin/dpd_ascendc_demo --help
    if [ $? -eq 0 ]; then
        echo "✅ 测试通过"
    else
        echo "❌ 测试失败"
    fi
else
    echo "⚠ 可执行文件未找到"
fi

echo -e "\n部署验证完成"
