# 机器学习分子动力学（Machine Learning Molecular Dynamics, ML-MD）

## 领域简介

分子动力学（MD）模拟是材料科学和化学工程中研究原子尺度动态行为的关键工具。传统 MD 使用经典力场（如 GAFF、OPLS、CHARMM）近似原子间相互作用，计算速度快但精度有限；第一性原理分子动力学（AIMD）精度高但计算成本极高，难以处理大规模体系（> 1000 原子）和长时间（> 1 ns）模拟。

机器学习分子动力学（ML-MD）在此背景下应运而生：用神经网络学习 DFT 级别的势能面（Potential Energy Surface, PES），在推理时达到与经典力场相当的计算速度，同时保持接近 DFT 的精度。本仓库现有算子覆盖了传统经典力场加速（Lennard-Jones、GAFF2、PME、SHAKE、Velocity Verlet），本方向则是向 ML 势扩展，构建 "经典力场 + ML 势 + 主动学习" 的完整 MD 加速链路。

## 关键模型与算法

### DeepMD

DeepMD（Deep Potential）是目前工业界部署最广的 ML 势方法：

- **架构**：Embedding net（编码原子种类和局部环境）→ Fitting net（映射到能量和力）
- **核心算子**：embedding 矩阵构建、网络前向传播（矩阵乘法 + 激活函数）
- **优势**：支持大体系（百万原子级别）的并行 MD 模拟，已在 GROMACS、LAMMPS 中集成
- **代表工作**：Zhang et al., *Phys. Rev. Lett.* 2018; *Comput. Phys. Commun.* 2018。
- **开源实现**：https://github.com/deepmodeling/deepmd-kit

### SchNet / PaiNN

基于连续滤波器卷积的等变图神经网络：

- **SchNet**：利用距离基函数（Gaussian basis）和连续滤波器实现坐标间的等变消息传递。Schütt et al., *NeurIPS* 2017。
- **PaiNN**：在 SchNet 基础上引入极性等变（equivariant vector features），支持偶极矩/极化率预测。Schütt et al., *NeurIPS* 2021。

### MACE / NequIP

多体等变图神经网络——目前精度最高的 ML 势方法之一：

- **核心理念**：利用不可约张量积构造等变（equivariant）消息，显式编码多体相互作用
- **MACE**：将消息用球谐函数展开，在消息传递过程中保持旋转等变性。Batatia et al., *NeurIPS* 2022。
- **NequIP**：基于 Tensor Product 的等变神经网络。Batzner et al., *Nat. Commun.* 2022。
- **算子化难点**：球形谐波（Spherical Harmonics）和 Clebsch-Gordan 系数的加速计算。

### ANI

原子中心神经网络（AENet）：

- 每个原子的能量由其局域原子环境向量（AEV）独立决定，求和得总能量
- **AEV**：以原子为中心的径向和角度对称函数组
- **优势**：推理效率高，适合高通量筛选
- **代表工作**：Smith et al., *Chem. Sci.* 2017。

### 在线学习与主动学习

ML-MD 中的关键工程技术：

- **描述符在线计算**：在 MD 轨迹运行时实时计算 SOAP（Smooth Overlap of Atomic Positions）、ACSF（Atom-Centered Symmetry Functions）、Behler-Parrinello 对称函数
- **不确定性探测**：利用 ensemble 方差、sGP（稀疏高斯过程）或贝叶斯 dropout 估计预测不确定性
- **On-the-fly 训练**：发现高不确定性结构时，调用 DFT 计算补充训练集，更新势能面

### D3 色散校正

Grimme D3 / D4 色散校正作为 ML 势的物理先验修正：

- 基于原子坐标计算两体（$C_6$）和三体（$C_9$）色散项
- 阻尼函数（Becke-Johnson / Zero-damping）在短程区域平滑截断
- 可叠加到任意 ML 势上提高含色散体系的精度

## 典型化工应用

| 应用场景 | 推荐方法 | 说明 |
|----------|----------|------|
| 催化剂表面吸附/反应 | DeepMD / MACE | 数千原子级别催化界面模拟，预测吸附能和反应路径 |
| 高分子熔体流变 | SchNet / DeepMD | 长链聚合物在剪切场中的构象变化与黏度预测 |
| 电池电解质离子传输 | PaiNN / NequIP | 考虑极性环境的离子迁移率与电导率预测 |
| 气体分离膜渗透 | ANI / DeepMD | 气体分子在膜材料中的扩散系数和渗透率 |
| 纳米材料力学 | MACE | 纳米颗粒/纳米管在拉伸压缩下的力学响应 |

## 本仓库现有基础

本仓库已在 `simulation/AI4MD/` 目录下实现了以下经典力场 MD 算子：

| 算子 | 功能 | 状态 |
|------|------|------|
| **Lennard-Jones** | LJ 12-6 力场融合计算 | ✅ 已发布 |
| **GAFF2** | GAFF2 力场（键/角/二面角/LJ/库仑） | ✅ 已发布 |
| **PME** | Ewald 求和 + PME 加速长程静电 | ✅ 已发布 |
| **SHAKE** | 迭代键长约束 | ✅ 已发布 |
| **Velocity Verlet** | 3 步时间积分器 | ✅ 已发布 |
| **DPD** | 耗散粒子动力学 | ✅ 已发布 |

本方向是在经典力场基础上的 ML 势扩展，规划中的算子包括：

- DeepMD / SchNet / PaiNN / MACE / ANI 等势能推理算子
- 描述符在线计算算子（SOAP / ACSF）
- D3/D4 色散校正算子
- 势能面在线拟合算子（sGP / 贝叶斯线性回归）

> 详细规划请参见 [roadmap.md](../../roadmap.md)。