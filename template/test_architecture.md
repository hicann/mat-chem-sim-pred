# {算子名称} — 测试架构模板

> 说明该算子的测试策略、目录结构和编写规范。目标是确保算子正确性、精度和性能满足上线要求。

---

## 测试层级

| 层级 | 文件位置 | 语言 | 测试内容 |
|------|----------|------|----------|
| **C++ 单元测试** | `tests/ut/op_kernel/test_{算子名}.cpp` | C++ | Kernel 级正确性验证，与 CPU 参考实现逐结果对比 |
| **Python 集成测试** | `tests/test_{算子名}.py` | Python | 端到端调用验证，对比 NumPy/PyTorch 参考实现 |
| **性能基准测试** | `tests/benchmark_{算子名}.py` | Python | 多规模性能数据采集，计算加速比 |

## 测试目录结构

```
tests/
├── CMakeLists.txt              # 递归包含所有子目录
├── test_{算子名}.py             # Python 集成测试
├── benchmark_{算子名}.py        # 性能基准测试（可选）
└── ut/
    ├── CMakeLists.txt           # 递归包含子目录
    └── op_kernel/
        ├── CMakeLists.txt       # 注册 UT 测试用例
        └── test_{算子名}.cpp     # C++ 单元测试
```

## C++ 单元测试

### 测试结构

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 
 * \file test_{算子名}.cpp
 * \brief {算子名称} 算子完整测试 (精度 + 性能)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"

// 声明算子 API
extern "C" int32_t aclnn{算子名}(...);
extern "C" uint64_t aclnn{算子名}GetWorkspaceSize(...);

// CPU 参考实现
void {算子名}CPU(const std::vector<float>& input,
                std::vector<float>& output) {
    // 用纯 C++ 实现参考计算逻辑
}

// 测试用例 1: 小规模正确性验证
void TestSmallScale() {
    // 构造输入
    // 分别在 CPU 和 NPU 上执行
    // 逐元素对比，计算最大绝对误差和相对误差
}

// 测试用例 2: 边界条件
void TestBoundaryConditions() {
    // 测试零输入、最大输入、截断条件等边界情况
}

// 测试用例 3: 物理守恒律
void TestConservationLaws() {
    // 如牛顿第三定律（作用力=反作用力）、能量守恒等
}

// 测试用例 4: 性能测试
void TestPerformance() {
    // 多规模计时，输出耗时和加速比
}
```

### CMake 注册

```cmake
# tests/ut/op_kernel/CMakeLists.txt
if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase({算子名} "ascend910B1" "" )
endif()
```

## Python 集成测试

### 测试结构

```python
#!/usr/bin/env python3
"""
{算子名称} - Unit Tests
"""

import numpy as np
import pytest

# NumPy 参考实现
def {算子名}_numpy(params) -> np.ndarray:
    """用 NumPy 实现参考计算逻辑"""
    pass

# PyTorch 参考实现（可选）
def {算子名}_pytorch(params):
    """用 PyTorch 实现参考计算逻辑"""
    pass

class Test{算子名}:
    """{算子名称} 算子测试类"""

    def test_small_case(self):
        """小规模正确性验证"""
        result_npu = ...    # 调用 NPU 算子
        result_cpu = {算子名}_numpy(...)
        assert np.allclose(result_npu, result_cpu, atol=1e-5)

    def test_boundary(self):
        """边界条件测试"""
        pass

    def test_numerical_accuracy(self):
        """数值精度测试，与 CPU double 参考值对比"""
        pass
```

### Pytest 运行

```bash
# 运行所有测试
pytest tests/test_{算子名}.py -v

# 运行单个测试用例
pytest tests/test_{算子名}.py::Test{算子名}::test_small_case -v
```

## 验证指标

| 指标 | 要求 | 说明 |
|------|------|------|
| **数值精度** | 相对误差 ≤ 1×10⁻³ | 与 CPU double 参考实现对比 |
| **物理守恒** | 守恒律偏差 ≤ 1×10⁻⁵ | 如动量守恒、能量守恒等 |
| **加速比** | ≥ 10x (vs CPU, 大规模) | 在典型工业规模输入上测量 |
| **收敛性** | 满足算法理论收敛阶 | 如迭代法在指定步数内收敛 |

## 测试数据管理

- 小规模测试数据直接在代码中构造
- 大规模测试数据使用 `tests/ut/op_kernel/{算子名}_data/gen_data.py` 生成
- 基准数据存放在 `tests/ut/op_kernel/{算子名}_data/` 目录下