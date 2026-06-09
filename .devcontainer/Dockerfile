# =============================================================================
# mat-chem-sim-pred — CANN Ascend C 算子开发 Docker 镜像
# =============================================================================
# 构建方式:
#   docker build -t mat-chem-sim-pred-dev .
#
# 运行方式 (有 NPU 设备):
#   docker run --rm -it \
#     --device=/dev/davinci0 \
#     --device=/dev/davinci_manager \
#     --device=/dev/hisi_hdc \
#     -v /usr/local/dcmi:/usr/local/dcmi \
#     -v /var/log/npu/:/var/log/npu \
#     -v $(pwd):/workspace \
#     mat-chem-sim-pred-dev
#
# 运行方式 (无 NPU 设备，仅 PyTorch 开发):
#   docker run --rm -it -v $(pwd):/workspace mat-chem-sim-pred-dev
# =============================================================================

# === 阶段一: CANN 基础镜像 ====================================================
# 如果使用华为云 SWR 的 CANN 官方镜像:
# FROM swr.cn-north-4.myhuaweicloud.com/ascend/cann:7.0.RC1-ubuntu22.04-aarch64
#
# 如果无 CANN 镜像访问权限，使用 Ubuntu 基础镜像并手动安装 CANN:

FROM ubuntu:22.04 AS base

LABEL maintainer="mat-chem-sim-pred team"
LABEL description="Development environment for Ascend C operators in mat-chem-sim-pred"

# 避免交互式配置
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 基础工具
    git \
    curl \
    wget \
    vim \
    ca-certificates \
    # 编译工具
    cmake \
    build-essential \
    gcc \
    g++ \
    make \
    # Python
    python3 \
    python3-pip \
    python3-venv \
    # 调试工具
    gdb \
    valgrind \
    # 数学库
    libfftw3-dev \
    libopenblas-dev \
    # 清理
    && rm -rf /var/lib/apt/lists/*

# 创建 Python 软链接
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 安装 Python 包
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    torch \
    numpy \
    matplotlib \
    pandas \
    pytest \
    jupyter \
    ase==3.23.0 \
    && rm -rf ~/.cache/pip

# =============================================================================
# CANN Toolkit 安装 (二选一)
# =============================================================================
#
# 方式 A: 从华为云 SWR 拉取 CANN 官方镜像 (推荐)
#   参见 https://www.hiascend.com/developer/devkit 获取镜像地址
#
# 方式 B: 手动安装 CANN (使用本 Dockerfile)
#   1. 从 https://www.hiascend.com/software/cann/community 下载 ASCEND_TOOLKIT
#   2. 将安装包放在同目录下，取消下面注释并修改版本号
#
# ARG CANN_VERSION=7.0.RC1
# COPY Ascend-cann-toolkit_${CANN_VERSION}_linux-aarch64.run /tmp/
# RUN chmod +x /tmp/Ascend-cann-toolkit_${CANN_VERSION}_linux-aarch64.run && \
#     /tmp/Ascend-cann-toolkit_${CANN_VERSION}_linux-aarch64.run --quiet --install \
#         --install-path=/usr/local/Ascend && \
#     rm /tmp/Ascend-cann-toolkit_${CANN_VERSION}_linux-aarch64.run

# === 阶段二: 开发者镜像 =======================================================
FROM base AS dev

# 设置 CANN 环境变量 (仅在安装 CANN 后生效)
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/lib:$LD_LIBRARY_PATH
ENV PATH=$ASCEND_TOOLKIT_HOME/bin:$ASCEND_TOOLKIT_HOME/compiler/bin:$PATH

# 设置工作目录
WORKDIR /workspace

# 安装 pre-commit 钩子
RUN pip3 install pre-commit

# 默认命令
CMD ["/bin/bash"]