#!/bin/bash
# =============================================================================
# postCreateCommand: 容器首次启动后的自动化环境配置
# =============================================================================
set -e

echo "============================================"
echo "  mat-chem-sim-pred 开发环境配置"
echo "============================================"

# 1. 验证基础环境
echo ""
echo "[1/5] 验证基础环境..."
python3 --version
cmake --version | head -1
gcc --version | head -1
pip3 --version

# 2. 验证 PyTorch
echo ""
echo "[2/5] 验证 PyTorch..."
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'cann'):
    print(f'CANN available: {torch.backends.cann.is_available()}')
" 2>&1 || echo "⚠ PyTorch 验证未通过 (不影响 Ascend C 编译)"

# 3. 验证 CANN 环境 (如有)
echo ""
echo "[3/5] 验证 CANN 环境..."
if [ -d "$ASCEND_TOOLKIT_HOME" ]; then
    echo "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME ✅"
    ls -la $ASCEND_TOOLKIT_HOME/compiler/bin/*cc 2>/dev/null || echo "⚠ 编译器未找到 (可能需设置交叉编译)"
else
    echo "⚠ ASCEND_TOOLKIT_HOME 未设置或路径不存在"
    echo "   Ascend C 算子编译需要 CANN Toolkit"
    echo "   可从 https://www.hiascend.com/software/cann/community 下载"
fi

# 4. 安装 pre-commit 钩子
echo ""
echo "[4/5] 安装 pre-commit 钩子..."
cd /workspace
if [ -f .pre-commit-config.yaml ]; then
    pre-commit install --install-hooks 2>&1 || echo "⚠ pre-commit 安装跳过"
    echo "   pre-commit 钩子已安装 ✅"
else
    echo "   .pre-commit-config.yaml 不存在，跳过"
fi

# 5. 运行环境冒烟测试
echo ""
echo "[5/5] 环境冒烟测试..."
echo ""

# 测试 PyTorch 基础运算
python3 -c "
import torch, numpy as np
a = torch.randn(10, 10)
b = torch.randn(10, 10)
c = a @ b
print(f'PyTorch 矩阵乘法: {a.shape} @ {b.shape} = {c.shape} ✅')

# 测试 NumPy
arr = np.array([1, 2, 3])
print(f'NumPy 基础运算: {arr.sum()} ✅')
" 2>&1 || echo "⚠ 冒烟测试部分失败"

# 测试仓库导入
python3 -c "
import sys, os
sys.path.insert(0, '/workspace')
from prediction.TabularData.tabnet import TabNetEncoder
print(f'TabNet 导入成功 ✅')

from prediction.TimeSeries.timesnet import TimesNet
print(f'TimesNet 导入成功 ✅')

from prediction.SmallData.kernels import RBFKernel
print(f'GPR kernels 导入成功 ✅')
" 2>&1 || echo "⚠ 部分模块导入失败"

echo ""
echo "============================================"
echo "  环境配置完成"
echo "============================================"
echo ""
echo "快速命令:"
echo "  - 构建 LJ 算子: cd simulation/AI4MD/Lennard_Jones && bash build.sh"
echo "  - 运行测试:     cd build && ctest --output-on-failure"
echo "  - 启动 Jupyter:  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser"
echo "============================================"