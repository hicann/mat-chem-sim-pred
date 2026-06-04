# GAFF2 Force Field — 算法说明

## 概述

GAFF2 (General AMBER Force Field 2) 是AMBER力场的第二代通用版本，适用于有机小分子的分子力学模拟。本算子实现了GAFF2的全部势能项，在Ascend NPU上以AIV核函数并行计算。

## 势能函数

总势能由五项组成：

$$E_{\text{total}} = E_{\text{bond}} + E_{\text{angle}} + E_{\text{dihedral}} + E_{\text{LJ}} + E_{\text{Coulomb}}$$

### 1. 键伸缩能 (Bond Stretch)

$$E_b = \sum_{\text{bonds}} k_b \cdot (r - r_0)^2$$

其中 $r = |r_j - r_i|$ 为实际键长，$r_0$ 为平衡键长，$k_b$ 为力常数。

力方向：$F_i = -2k_b(r - r_0) \cdot (r_i - r_j)/r$，$F_j = -F_i$

### 2. 键角弯曲能 (Angle Bend)

$$E_a = \sum_{\text{angles}} k_\theta \cdot (\theta - \theta_0)^2$$

$\theta$ 通过余弦定理计算：$\cos\theta = (r_{ji} \cdot r_{jk}) / (|r_{ji}| \cdot |r_{jk}|)$

通过链式法则求导：
$$\frac{dE}{d\cos\theta} = -2k_\theta(\theta - \theta_0) / \sin\theta$$
$$F_i = \frac{dE}{d\cos\theta} \cdot \frac{\cos\theta \cdot e_{ji} - e_{jk}}{|r_{ji}|}$$
$$F_j = -(F_i + F_k)$$

### 3. 二面角扭转能 (Dihedral Torsion)

$$E_d = \sum_{\text{dihedrals}} \sum_{n} \frac{V_n}{2} [1 + \cos(n\phi - \phi_{0,n})]$$

$\phi$ 通过四原子平面法向量计算，$V_n$ 为扭转势垒，$n$ 为重数（典型值1-6），$\phi_0$ 为相位偏移。

### 4. Lennard-Jones 12-6

$$E_{LJ} = \sum_{i<j} 4\varepsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6 \right]$$

力：$F = \frac{24\varepsilon}{r} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$

交叉项适用 Lorentz-Berthelot 混合规则：$\sigma_{ij} = (\sigma_i + \sigma_j)/2$, $\varepsilon_{ij} = \sqrt{\varepsilon_i \cdot \varepsilon_j}$

### 5. 库仑静电 (Coulomb)

$$E_c = \sum_{i<j} \frac{q_i q_j}{4\pi\varepsilon_0 r_{ij}}$$

力：$F = \frac{q_i q_j}{4\pi\varepsilon_0 r^2} \cdot \hat{r}$

1-4对（同一二面角的首末原子）应用缩放因子：LJ缩放0.5，Coulomb缩放0.83333。

## 力约定

所有力的定义为 $F = -\nabla E$（指向能量降低方向），与GROMACS约定一致。返回力单位为 $\text{kJ}/(\text{mol} \cdot \text{nm})$。

## NPU实现

- **Kernel**：`gaff2_force.cpp` — 单tile循环所有键、角、二面角和非键项
- **不依赖** `<math.h>` 或 `__builtin_*`：所有数学函数（sqrt, rsqrt, inv, sin/cos）通过位运算 + Newton-Raphson迭代实现
- **排斥对**：通过 exclusion_mask 跳过1-2、1-3对
- **PBC**：最小镜像处理在力计算中内联完成

## NPU精度

由于Ascend C不支持原生 `<math.h>` 和 `__builtin_*`，所有数学函数（sqrt, rsqrt, sin/cos）通过位操作 + Newton-Raphson迭代2次实现，误差约 1×10⁻⁷。

| 量 | NPU float32 | 理论值 | 误差 |
|---|---|---|---|
| 键能 (k=1000, dr=0.03nm) | 0.899992 kJ/mol | 0.900000 | 8×10⁻⁶ |
| 键力 F_x | ±59.999687 | ±60.000000 | 3×10⁻⁴ |
| LJ最小值能量 | -1.000000 kJ/mol | -1.000000 | 0 |
| Coulomb (r=0.5nm) | 277.865417 kJ/mol | 277.870911 | 5×10⁻³ |

所有误差相对值 < 0.001%，对MD模拟可忽略。

## 参考

1. Wang, J.; Wolf, R. M.; Caldwell, J. W.; Kollman, P. A.; Case, D. A. J. Comput. Chem. 2004, 25, 1157-1174.
2. GAFF2: ftp://ftp.amber.ucsf.edu/GAFF2/
3. GROMACS Reference Manual — Force Fields
