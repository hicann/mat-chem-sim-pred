# Velocity Verlet 积分器 — 算法说明

## 概述

Velocity Verlet (VV) 是分子动力学中最广泛使用的时间积分算法。它在时间反演对称性和辛结构上优于传统的Runge-Kutta方法，是长时间MD模拟的标准选择。

## 算法

### 标准3步形式

对于运动方程 $m_i \ddot{r}_i = F_i(r)$：

```
Step A (half-kick):  v(t+dt/2) = v(t) + (dt/2) · a(t)
Step B (drift):      r(t+dt)   = r(t) + dt · v(t+dt/2)
Step C (half-kick):  v(t+dt)   = v(t+dt/2) + (dt/2) · a(t+dt)
```

其中 $a(t) = F(r(t))/m$。

### 本实现的两核设计

由于Ascend C核函数间无法直接共享中间数据，积分器拆分为两个独立核：

| Kernel | 功能 | 执行内容 |
|--------|------|----------|
| `vv_integrate` | Step A + B | $v(t+dt/2)$ 更新 → $r(t+dt)$ 更新（含PBC折叠） |
| `vv_finish` | Step C | $v(t+dt)$ 更新 |

中间步骤（力评价）由Host侧编排：

```
Host:  Launch vv_integrate → 等待 → Host调用ForceEval → Launch vv_finish
```

## NPU核函数

### vv_integrate
```
输入: v(t), r(t), F(t), masses, dt
输出: v(t+dt/2), r(t+dt)

forEach atom i:
    a_i = F_i / m_i                    // 加速度
    v_i_new = v_i + half_dt * a_i      // 半步步
    r_i_new = r_i + dt * v_i_new       // 位置更新
    if PBC: r_i_new = fold(r_i_new)    // 周期边界折叠
```

### vv_finish
```
输入: v(t+dt/2), F(t+dt), masses, half_dt
输出: v(t+dt)

forEach atom i:
    a_i_new = F_i_new / m_i
    v_i_new = v_i + half_dt * a_i_new
```

## 数学性质

- **时间反演对称**：$\Theta VV = I$（精确互逆）
- **辛结构**：保持相空间体积不变
- **局域误差**：$O(dt^3)$
- **全局误差**：$O(dt^2)$
- **能量漂移**：随步长$O(dt^2)$（有界）

## NPU精度

NPU单精度float32与CPU双精度double路径对比（5步VV）：

| 量 | 最大差异 |
|---|---|
| 坐标差 | 5.86×10⁻⁷ nm |
| 速度差 | 1.91×10⁻⁸ nm/ps |

差异 < 1×10⁻⁶ nm，远低于典型MD精度要求（~0.001nm）。

## 参考

1. Swope, W. C.; Andersen, H. C.; Berens, P. H.; Wilson, K. R. J. Chem. Phys. 1982, 76, 637-649.
2. Verlet, L. Phys. Rev. 1967, 159, 98-103.
3. Tuckerman, M. E. Statistical Mechanics: Theory and Molecular Simulation; Oxford, 2010.
