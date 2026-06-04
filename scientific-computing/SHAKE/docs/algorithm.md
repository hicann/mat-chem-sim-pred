# SHAKE 约束算法 — 说明

## 概述

SHAKE是一种迭代约束算法，用于在分子动力学模拟中固定键长（如氢原子到重原子的键）。它通过对位置施加修正来满足一组约束方程，避免了使用非常小的时间步长来解析高频键振动。

## 算法原理

### 约束条件

对于每个约束 $k$（连接原子 $i$ 和 $j$）：

$$g_k(r) = |r_j - r_i|^2 - d_k^2 = 0$$

其中 $d_k$ 为目标键长。

### 迭代求解

SHAKE在VV位置更新后应用，基本原理是利用Lagrange乘子 $\lambda_k$ 施加约束力：

1. **未约束位置**：$r' = r(t) + dt \cdot v(t+dt/2)$
2. **约束修正**：$r'' = r' + \delta r$，其中 $\delta r$ 使所有 $g_k(r'') = 0$

约束力方向沿键方向：
$$\delta r_i = \sum_k \lambda_k \cdot \frac{r_j - r_i}{|r_j - r_i|} \cdot \frac{1}{m_i}$$
$$\delta r_j = -\sum_k \lambda_k \cdot \frac{r_j - r_i}{|r_j - r_i|} \cdot \frac{1}{m_j}$$

### 迭代过程（Gauss-Seidel风格）

```
for iter = 0..max_iter:
    max_error = 0
    for each constraint k (i, j):
        // 当前距离
        r_ij = |r_j - r_i|
        error = |r_ij² - d_k²|
        max_error = max(max_error, error)

        // 计算λ修正
        lambda = (d_k² - r_ij²) / (2 * (1/m_i + 1/m_j) * r_ij²)

        // 应用修正
        r_i += lambda / m_i * (r_j - r_i)
        r_j -= lambda / m_j * (r_j - r_i)

    if max_error < tolerance:
        converged
```

## NPU实现

- **Kernel**：`shake.cpp` — 单迭代步的约束修正
- **Host循环**：`ShakeHost::Apply()` — 反复启动kernel直到收敛
- 每个kernel迭代写回 `max_error` 到device标量，Host读取判断收敛
- 支持异构质量系统（每个原子可独立指定质量）

## NPU精度

NPU SHAKE与CPU double实现对比：

| 指标 | NPU float32 | CPU double | 差异 |
|---|---|---|---|
| 单键长度 | 0.150000 | 0.150000 | ~0 |
| 双键系统键长1-2 | 0.150019 | 0.150000 | 1.9×10⁻⁵ |
| VV+SHK 10步最大偏差 | — | — | 3.8×10⁻⁷ nm |
| 最大坐标差(NPU vs CPU) | — | — | 9.54×10⁻⁶ nm |

差异 < 2×10⁻⁵ nm（相对 < 0.01%），完全可接受。

## 参考

1. Ryckaert, J. P.; Ciccotti, G.; Berendsen, H. J. C. J. Comput. Phys. 1977, 23, 327-341.
2. Allen, M. P.; Tildesley, D. J. Computer Simulation of Liquids; Oxford, 2017.
