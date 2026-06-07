# DPD算子设计文档

## 概述

本文档描述耗散粒子动力学（Dissipative Particle Dynamics, DPD）Ascend C算子的整体设计，包括算法原理、硬件适配、优化策略和实现细节。

## 1. 算法原理

### 1.1 DPD基本方程

DPD是一种介观粗粒化粒子方法，每个粒子代表一组分子或流体微团。系统演化遵循牛顿运动方程：

**运动方程**:
```
dr_i/dt = v_i
dv_i/dt = F_i / m_i
```

**总作用力**（对每一对粒子 i-j）:
```
F_ij = F_C + F_D + F_R
```

#### 1.1.1 保守力（Conservative Force）
```
F_C = a_ij × (1 - r/rc) × e_ij
```
- `a_ij`: 粒子间相互作用强度
- `r = |r_ij|`: 粒子间距离
- `rc`: 截断半径
- `e_ij = r_ij / r`: 单位方向向量

#### 1.1.2 耗散力（Dissipative Force）
```
F_D = -γ × w_D(r)² × (e_ij · v_ij) × e_ij
```
- `γ`: 耗散系数
- `v_ij = v_i - v_j`: 相对速度
- `w_D(r) = 1 - r/rc`: 耗散权重函数

#### 1.1.3 随机力（Random Force）
```
F_R = σ × w_R(r) × ξ_ij × e_ij
```
- `σ`: 噪声强度
- `ξ_ij`: 高斯随机数（均值为0，方差为1）
- `w_R(r) = 1 - r/rc`: 随机权重函数

#### 1.1.4 涨落-耗散定理
```
σ² = 2γ k_B T
```
- `k_B T`: 热浴能量（通常取1进行无量纲模拟）

### 1.2 运动积分（Velocity-Verlet算法）

每个时间步 `dt` 分为两步：

**第一步（预测步）**:
```
v_i(t + dt/2) = v_i(t) + (F_i(t) / m_i) × (dt/2)
r_i(t + dt) = r_i(t) + v_i(t + dt/2) × dt
```

**第二步（校正步）**:
```
根据新位置 r_i(t+dt) 重新计算所有力 F_i(t+dt)
v_i(t + dt) = v_i(t + dt/2) + (F_i(t+dt) / m_i) × (dt/2)
```

### 1.3 周期性边界条件（PBC）

```
r_ij = r_ij - round(r_ij / L) × L
```
- `L`: 模拟盒子边长

### 1.4 邻居搜索

使用细胞列表（Cell List）加速邻居搜索：
- 将模拟空间划分为大小为 `rc` 的网格
- 每个粒子分配到对应的细胞
- 只搜索相邻细胞中的粒子对

## 2. 硬件适配设计

### 2.1 AI Core架构适配

#### 2.1.1 向量化计算
- 使用Vector单元进行SIMD计算
- 粒子数据按16个一组进行向量化处理
- 力计算使用 `vec_fma` 等向量指令

#### 2.1.2 内存层次优化
```
Global Memory (GM) → Local Memory (L1) → Unified Buffer (UB)
```
- **GM**: 存储所有粒子数据
- **L1**: 缓存频繁访问的邻居数据
- **UB**: 存储当前计算的粒子块

#### 2.1.3 计算流水线
```
数据搬运 (MTE) → 邻居搜索 → 力计算 → 运动积分 → 数据写回
```

### 2.2 Tiling策略

#### 2.2.1 粒子分块
```
总粒子数 N → 分块大小 B → 块数 K = ceil(N/B)
```
- 每个AI Core处理一个或多个粒子块
- 块大小根据UB容量优化

#### 2.2.2 邻居列表分块
- 为每个粒子块预计算邻居列表
- 邻居数据存储在L1缓存中
- 使用双缓冲隐藏数据搬运延迟

### 2.3 并行策略

#### 2.3.1 粒子级并行
- 不同粒子分配到不同AI Core
- 负载均衡：根据邻居数量动态分配

#### 2.3.2 力计算并行
- 粒子对计算并行化
- 使用原子操作更新力（避免竞争）

## 3. 优化策略

### 3.1 内存访问优化

#### 3.1.1 数据布局
```cpp
// 结构体数组布局 (AoS)
struct Particle {
    float x, y, z;     // 位置
    float vx, vy, vz;  // 速度
    float fx, fy, fz;  // 力
    float type;        // 粒子类型
};

// 数组结构布局 (SoA) - 向量化友好
struct ParticleData {
    float* x, *y, *z;
    float* vx, *vy, *vz;
    float* fx, *fy, *fz;
    float* type;
};
```

#### 3.1.2 内存对齐
- 粒子数据32字节对齐
- 向量化访问128字节对齐
- 使用 `alignas(32)` 确保对齐

### 3.2 计算优化

#### 3.2.1 近似计算
- 使用快速平方根倒数 `rsqrt`
- 查表法计算权重函数
- 低精度计算（fp16）可选

#### 3.2.2 分支优化
- 使用掩码操作避免分支
- 预计算截断半径判断
- 向量化条件选择

### 3.3 通信优化

#### 3.3.1 减少同步
- 异步数据搬运
- 双缓冲技术
- 流水线执行

#### 3.3.2 数据局部性
- 时间局部性：重用邻居列表
- 空间局部性：细胞列表优化

## 4. 实现架构

### 4.1 目录结构
```
project_root/
├── op_kernel/           # 内核端实现
│   ├── dpd_kernel.cpp   # 核函数主逻辑
│   ├── dpd_tiling.cpp   # Tiling策略
│   ├── dpd_kernel.h     # 内核函数声明
│   └── dpd_tiling.h     # Tiling结构定义
├── op_host/             # 主机端实现
│   ├── dpd_host.cpp     # 主机调度逻辑
│   ├── dpd_host.h       # 主机接口声明
│   └── dpd_params.h     # 参数定义
├── tests/               # 测试代码
├── examples/            # 示例代码
├── docs/                # 文档
└── op_proto/            # 算子原型
```

### 4.2 内核设计

#### 4.2.1 核函数签名
```cpp
extern "C" __global__ __aicore__ void dpd_kernel(
    uint8_t* pos_gm,        // 位置数据
    uint8_t* vel_gm,        // 速度数据
    uint8_t* force_gm,      // 力数据
    uint8_t* params_gm,     // 模拟参数
    uint8_t* tiling_gm      // Tiling参数
);
```

#### 4.2.2 计算流程
1. **数据加载**: GM → UB（双缓冲）
2. **邻居搜索**: 细胞列表构建
3. **力计算**: 保守力 + 耗散力 + 随机力
4. **运动积分**: Velocity-Verlet算法
5. **边界处理**: 周期性边界条件
6. **数据写回**: UB → GM

### 4.3 主机端设计

#### 4.3.1 接口设计
```cpp
class DpdSimulator {
public:
    bool initialize(const DpdParams& params);
    DpdResult run_simulation();
    DpdResult run_step(int32_t num_steps);
    // ... 其他接口
};
```

#### 4.3.2 内存管理
- 设备内存分配/释放
- 数据主机-设备传输
- 错误处理和恢复

## 5. 性能分析

### 5.1 计算复杂度
- **邻居搜索**: O(N × M)，M为平均邻居数
- **力计算**: O(N × M)
- **运动积分**: O(N)

### 5.2 内存需求
```
总内存 = 粒子数据 + 邻居列表 + 临时缓冲区
粒子数据 = N × (3位置 + 3速度 + 3力 + 1类型) × sizeof(float)
邻居列表 ≈ N × M × (索引 + 距离)
```

### 5.3 性能指标
- **计算吞吐量**: 粒子对/秒
- **内存带宽**: GB/秒
- **能效**: 粒子对/焦耳

## 6. 扩展性设计

### 6.1 多精度支持
- fp32: 标准精度
- fp16: 混合精度训练
- bf16: 未来扩展

### 6.2 多GPU支持
- 空间分解
- 粒子迁移
- 边界通信

### 6.3 可配置参数
- 力场参数
- 积分算法
- 边界条件
- 输出频率

## 7. 验证策略

### 7.1 单元测试
- 力计算正确性
- 运动积分精度
- 边界条件处理

### 7.2 集成测试
- 完整模拟流程
- 能量守恒验证
- 性能基准测试

### 7.3 回归测试
- 参考实现对比
- 精度容差检查
- 性能回归检测

## 8. 部署指南

### 8.1 环境要求
- **硬件**: 昇腾910/310P NPU
- **软件**: CANN工具包 5.0+
- **依赖**: PyTorch 1.8+（可选）

### 8.2 编译步骤
```bash
mkdir build && cd build
cmake .. -DCANN_PATH=/path/to/cann
make -j$(nproc)
```

### 8.3 运行示例
```bash
# C++示例
./bin/dpd_ascendc_demo

# Python示例
python examples/dpd_pytorch_demo.py
```

## 9. 维护和扩展

### 9.1 代码维护
- 遵循C++核心指南
- 完整的文档注释
- 自动化测试套件

### 9.2 功能扩展
- 新的力场模型
- 不同的积分算法
- 高级输出分析

### 9.3 性能优化
- 算法改进
- 硬件特性利用
- 编译器优化

## 10. 参考文献

1. Hoogerbrugge, P. J., & Koelman, J. M. V. A. (1992). Simulating microscopic hydrodynamic phenomena with dissipative particle dynamics.
2. Español, P., & Warren, P. (1995). Statistical mechanics of dissipative particle dynamics.
3. Groot, R. D., & Warren, P. B. (1997). Dissipative particle dynamics: Bridging the gap between atomistic and mesoscopic simulation.
4. 华为昇腾文档: Ascend C编程指南

---

**文档版本**: 1.0.0  
**最后更新**: 2026-04-06  
**作者**: DPD算子开发团队  
**状态**: 审核通过