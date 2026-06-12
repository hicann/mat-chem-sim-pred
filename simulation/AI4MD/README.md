# 机器学习分子动力学（Machine Learning Molecular Dynamics, ML-MD）

## 领域简介

分子动力学（MD）模拟是材料科学和化学工程中研究原子尺度动态行为的关键工具。传统 MD 使用经典力场（如 GAFF、OPLS、CHARMM）近似原子间相互作用，计算速度快但精度有限；第一性原理分子动力学（AIMD）精度高但计算成本极高，难以处理大规模体系（> 1000 原子）和长时间（> 1 ns）模拟。

机器学习分子动力学（ML-MD）在此背景下应运而生：用神经网络学习 DFT 级别的势能面（Potential Energy Surface, PES），在推理时达到与经典力场相当的计算速度，同时保持接近 DFT 的精度。本仓库现有算子覆盖了传统经典力场加速（Lennard-Jones、GAFF2、PME、SHAKE、Velocity Verlet），本方向则是向 ML 势扩展，构建 "经典力场 + ML 势 + 主动学习" 的完整 MD 加速链路。

## 关键模型与算法

### 一、非等变图神经网络（Scalar GNN）

#### DeepMD

DeepMD（Deep Potential）是目前工业界部署最广的 ML 势方法：

- **架构**：Embedding net（编码原子种类和局部环境）→ Fitting net（映射到能量和力）
- **核心算子**：embedding 矩阵构建、网络前向传播（矩阵乘法 + 激活函数）
- **优势**：支持大体系（百万原子级别）的并行 MD 模拟，已在 GROMACS、LAMMPS 中集成
- **代表工作**：Zhang et al., *Phys. Rev. Lett.* 2018; *Comput. Phys. Commun.* 2018。
- **开源实现**：https://github.com/deepmodeling/deepmd-kit

#### SchNet

基于连续滤波器卷积的 GNN——最早的距离基函数 + 连续滤波器卷积模型：

- 利用高斯距离基函数生成连续滤波器，通过逐元素乘法对邻居特征加权聚合
- 不含球谐函数，训练稳定，推理速度中等
- **代表工作**：Schütt et al., *NeurIPS* 2017; *J. Chem. Phys.* 2018。
- **开源实现**：https://github.com/atomistic-machine-learning/schnetpack

#### ANI

原子中心神经网络（AENet），极轻量的 ML 势：

- 每个原子的能量由其局域原子环境向量（AEV）独立决定，含径向和角度对称函数
- 推理速度在标量网络中名列前茅，适合高通量筛选
- **代表工作**：Smith et al., *Chem. Sci.* 2017。
- **开源实现**：https://github.com/aiqm/torchani

#### CHGNet

在 Materials Project 150 万+无机晶体结构上预训练的通用电荷感知 ML 势：

- 独家磁矩（magmom）预测头，处理磁性材料
- 在 LAMMPS 中可直接作为力场调用
- **代表工作**：Deng et al., *Nat. Mach. Intell.* 2023。
- **开源实现**：https://github.com/CederGroupHub/chgnet

#### M3GNet

引入显式三体相互作用（角度信息）的通用势能模型：

- 三体消息传递 + 球谐展开 + 多图架构
- 结构与组成联合预测，支持弛豫和 MD，无机晶体通用势主流选择
- **代表工作**：Chen & Ong, *Nat. Comput. Sci.* 2022。
- **开源实现**：https://github.com/materialsvirtuallab/m3gnet

#### DimeNet / DimeNet++

基于方向消息传递的 GNN——显式编码键角和二面角：

- 不仅编码原子间距离，还编码键角（三元组）和二面角（四元组）
- DimeNet++ 用高斯基展开替代球贝塞尔函数，大幅提升效率
- **代表工作**：Gasteiger et al., *ICLR* 2020; *NeurIPS* 2020 Workshop。
- **开源实现**：https://github.com/klicperajo/dimenet

### 二、等变图神经网络（Equivariant GNN）

#### PaiNN

SchNet 的等变扩展版，标量+向量双通道消息传递：

- 标量消息和向量消息通过线性层 + 向量-标量乘法耦合
- 原生支持偶极矩/极化率（向量属性）预测，不含球谐函数
- **代表工作**：Schütt et al., *ICML* 2021。
- **开源实现**：https://github.com/atomistic-machine-learning/schnetpack (v2.0+)

#### NequIP

基于 SO(3) 等变张量积的消息传递网络——数据效率极高的精度标杆：

- 邻居的不可约特征 + 球谐方向向量 + 可学习路径权重 → Clebsch-Gordan 张量积
- 1000 个 DFT 结构即可达到 1 meV/atom 精度
- **代表工作**：Batzner et al., *Nat. Commun.* 2022。
- **开源实现**：https://github.com/mir-group/nequip

#### MACE

多体原子簇展开与等变消息传递的统一框架：

- 将 ACE 的 body-order 展开和等变 GNN 的消息传递结合
- MACE-MP-0 是 Materials Project 官方推荐的通用势（覆盖 89 种元素）
- 可以看作 NequIP 的泛化版本
- **代表工作**：Batatia et al., *NeurIPS* 2022; *arXiv* 2024 (MACE-MP-0)。
- **开源实现**：https://github.com/ACEsuit/mace

#### Allegro

NequIP 的"严格局部"版本——推理效率最优的等变网络：

- 去除消息传递，通过局域张量积构造多体特征
- 每个原子独立计算，天然适合 Ascend SIMD 并行范式
- 在 LAMMPS 中以 pair_style 方式集成，支持大规模并行 MD
- **代表工作**：Musaelian et al., *Nat. Commun.* 2023。
- **开源实现**：https://github.com/mir-group/allegro

#### TensorNet

基于笛卡尔张量的等变网络——无需球谐函数的简洁方案：

- 原子特征建模为 3x3 笛卡尔张量，通过张量收缩和逐元素非线性实现消息传递
- 不涉及球谐函数和 CG 系数的复杂计算，是等变网络中 Ascend 迁移难度最低的方案
- **代表工作**：Simeon & De Fabritiis, *NeurIPS* 2023。
- **开源实现**：https://github.com/torchmd/torchmd-net

#### GemNet

DimeNet 的泛化版本，引入四体相互作用：

- 方向消息传递 + 二面角消息传递（四体信息）
- 在 OC20 催化应用基准上性能优异
- **代表工作**：Gasteiger et al., *NeurIPS* 2021; *TMLR* 2022 (GemNet-OC)。
- **开源实现**：https://github.com/TUM-DAML/gemnet_pytorch

#### SO3krates

将 Transformer 与 SO(3) 等变表示结合的模型：

- SO(3) 等变自注意力机制，query/key/value 带有不可约表示标签
- 适合非局域相互作用体系（长程静电、极化）
- **代表工作**：Frank et al., *Nat. Commun.* 2024。
- **开源实现**：https://github.com/thorbenfrank/SO3krates

#### REANN

递归嵌入原子神经网络——中国科大蒋彬课题组提出：

- 使用"递归嵌入"构造原子环境描述符，避免显式角度计算
- 嵌入密度迭代更新自动编码多体信息，在气相小分子反应上表现优异
- **代表工作**：Zhang et al., *J. Chem. Theory Comput.* 2020; *J. Phys. Chem. Lett.* 2022。

### 三、大规模预训练通用势（Foundation Models）

- **MatterSim**：微软 Graphormer 架构，覆盖元素周期表全表（原子序数 1-103）、温度 0-5000K、压力 0-1000 GPa。Yang et al., *Nature* 2024。
- **Orb**：Orbital Materials 的通用原子模拟基础模型，等变消息传递 + 双分支（invariant/equivariant）。Neumann et al., *arXiv* 2024。
- **SevenNet**：基于 NequIP 架构的通用势，并行主动学习策略扩展训练数据。Park et al., *J. Chem. Theory Comput.* 2024。
- **GRACE**：基于张量积等变消息传递的通用势，可配置 body-order 在精度和效率间灵活权衡。Batatia et al., *arXiv* 2022。

### 四、在线学习与主动学习

ML-MD 中的关键工程技术：

- **描述符在线计算**：在 MD 轨迹运行时实时计算 SOAP（Smooth Overlap of Atomic Positions）、ACSF（Atom-Centered Symmetry Functions）、Behler-Parrinello 对称函数
- **不确定性探测**：利用 ensemble 方差、sGP（稀疏高斯过程）或贝叶斯 dropout 估计预测不确定性
- **On-the-fly 训练**：发现高不确定性结构时，调用 DFT 计算补充训练集，更新势能面

### 五、D3/D4 色散校正

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
| 纳米材料力学 | MACE / Allegro | 纳米颗粒/纳米管在拉伸压缩下的力学响应 |
| 大规模高通量筛选 | ORB / TensorNet | 百万级结构的快速能量/力评估 |

## 综合对比矩阵

| 模型 | 等变性 | 球谐函数 | 消息传递 | 推理速度 | Ascend 适宜度 |
|------|--------|----------|----------|----------|--------------|
| **DeepMD** | 否 | 否 | 否（局域） | 快 | **高** |
| **SchNet** | 否 | 否 | 是 | 中 | **高** |
| **ANI** | 否 | 否 | 否（局域） | 最快 | **高** |
| **PaiNN** | 是（向量） | 否 | 是 | 较快 | **高** |
| **TensorNet** | 是（笛卡尔张量） | 否 | 是 | 较快 | **高** |
| **CHGNet** | 否 | 否 | 是 | 中 | **中** |
| **M3GNet** | 否 | 否 | 是 | 中 | **中** |
| **DimeNet** | 否（方向） | 否 | 是（方向） | 慢 | **中** |
| **NequIP** | 是（SO3） | 是 | 是 | 慢 | **中** |
| **MACE** | 是（SO3） | 是 | 是 | 慢 | **中** |
| **Allegro** | 是（SO3） | 是 | 否（局域） | 较快 | **中**（最优等变） |
| **GemNet** | 否（方向） | 是（低阶） | 是（方向+二面角） | 最慢 | **中-低** |
| **SO3krates** | 是（SO3） | 是 | 是（Attention） | 中 | **低-中** |
| **MatterSim** | 否 | 否 | 是（Attention） | 中 | **中** |
| **Orb** | 是（SO3） | 是 | 是 | 中 | **低-中** |

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

- DeepMD / SchNet / PaiNN / ANI / TensorNet / Allegro 等势能推理算子
- MACE / NequIP / DimeNet / GemNet 等高等变势能推理算子
- MatterSim / CHGNet / M3GNet 等通用预训练势推理算子
- 描述符在线计算算子（SOAP / ACSF）
- D3/D4 色散校正算子
- 势能面在线拟合算子（sGP / 贝叶斯线性回归）

> 详细规划与待迁移模型清单请参见 [roadmap.md](../../roadmap.md) 和 [todo-algo-model.md](../../todo-algo-model.md)。