# Ewald / PME 长程静电 — 算法说明

## 概述

在周期性边界条件下，库仑相互作用 $1/r$ 是条件收敛的，不能直接截断。Ewald求和将静电分解为实空间和倒易空间两部分，使两者都快速收敛。PME (Particle Mesh Ewald) 通过FFT进一步加速倒易空间计算。

## 1. Ewald 求和

### 分解

$$E_{\text{elec}} = E_{\text{real}} + E_{\text{recip}} + E_{\text{self}}$$

使用误差函数 $\text{erfc}(x) = 1 - \text{erf}(x)$ 将 $1/r$ 分裂为：

$$\frac{1}{r} = \frac{\text{erfc}(\alpha r)}{r} + \frac{\text{erf}(\alpha r)}{r}$$

- **实空间项**：短程、在截断半径内评价：$E_{\text{real}} = \frac{1}{2} \sum_{i\neq j} q_i q_j \frac{\text{erfc}(\alpha r_{ij})}{r_{ij}}$
- **倒易空间项**：平滑、在倒格子空间求和：$E_{\text{recip}} = \frac{1}{2V} \sum_{k\neq 0} \frac{4\pi}{|k|^2} e^{-|k|^2/4\alpha^2} |\rho(k)|^2$
- **自能项**：减去自身相互作用：$E_{\text{self}} = -\frac{\alpha}{\sqrt{\pi}} \sum_i q_i^2$

### 分裂参数

$\alpha$（Ewald分割参数）控制实空间与倒易空间的平衡：较大的 $\alpha$ 使实空间收敛更快但倒易空间需要更多波矢。

## 2. PME (Particle Mesh Ewald)

### 从Ewald到PME

PME将电荷分布到网格上，用3D FFT计算倒易空间和，将复杂度从 $O(N^2)$ 降到 $O(N\log N)$。

### 步骤

```
1. 电荷分配 (Charge Spreading):
   将原子电荷 q_i 通过B样条插值分配到 FFT网格 (n_x × n_y × n_z)
   
2. 3D FFT:
   对密度网格执行3D复数FFT → ρ(k)
   
3. 倒易空间能量:
   E_recip = (1/2V) · Σ_k (4π/k²)·e^{-k²/4α²} · |ρ(k)|²
   
4. 力计算:
   通过逆FFT将倒易空间力变换回实空间
   插值得到每个原子的受力
```

### B样条插值

使用3阶B样条（$n_{\text{order}}=3$）将点电荷平滑分配到网格点：

$$W_p(x) = \frac{1}{6} \begin{cases} (2-x)^3 & 0 \leq x < 1 \\ 4 - 6x^2 + 3x^3 & 1 \leq x < 2 \\ (x-4)^3 & 2 \leq x < 3 \end{cases}$$

其中 $x = |r_i - r_{\text{grid}}| / \Delta r$。

## NPU实现

| 组件 | 文件 | 说明 |
|------|------|------|
| Ewald实空间力 | `ewald_kernel.cpp` | erfc修正Coulomb力 |
| PME Spreading | `pme_spread.h` | B样条电荷分配（NPU核） |
| PME FFT | `pme_fft3d.h` | 3D FFT（NPU核） |
| PME能量/力 | `pme_kernel.cpp` | 倒易空间能量+力 |
| Host编排 | `pme_host2.cpp` | FFT调度 + 网格管理 |

## 参考

1. Ewald, P. P. Ann. Phys. 1921, 64, 253-287.
2. Darden, T.; York, D.; Pedersen, L. J. Chem. Phys. 1993, 98, 10089-10092.
3. Essmann, U.; Perera, L.; Berkowitz, M. L.; Darden, T.; Lee, H.; Pedersen, L. G. J. Chem. Phys. 1995, 103, 8577-8593.
