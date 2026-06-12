<!--
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# AI for PDE — 物理信息驱动与神经算子模型

## 领域简介

化工过程中的大量物理场模拟（流体力学、传热传质、结构力学、反应-扩散耦合）本质上是偏微分方程（PDE）的求解问题。传统方法（有限差分 FDM、有限体积 FVM、有限元 FEM）在网格离散与数值迭代上消耗大量计算资源，尤其在高维参数化、多物理场耦合和实时预测场景中存在瓶颈。

近年来，**AI for PDE**（或称科学机器学习、AI-driven simulation）兴起，核心理念是用神经网络直接学习 PDE 的解算子或作为物理约束的近似求解器，在保持一定精度的前提下将推理速度提升 1-3 个数量级。本方向涵盖物理信息神经网络（PINN）、神经算子（Neural Operator）、网格上的图神经网络、可微分 PDE 层等路线。

## 关键模型与算法

### 一、物理信息神经网络（PINN）及其变体

PINN 将 PDE 的残差作为损失函数的正则化项，在无标签或少量标签下训练神经网络满足物理规律：

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{PDE}}$$

其中 $\mathcal{L}_{\text{PDE}}$ 包含控制方程残差、边界条件和初始条件。推理时，网络直接输出场变量（速度场 $\mathbf{u}$、压力场 $p$、温度场 $T$、浓度场 $c$）。

- **原始 PINN**：全连接网络 + 自动微分实现 PDE 残差约束。Raissi et al., *J. Comput. Phys.* 2019。
- **cPINN**（Conservative PINN）：子域划分 + 界面通量连续性约束，适合守恒律 PDE。Jagtap et al., *Comput. Methods Appl. Mech. Eng.* 2020。
- **XPINN**（Extended PINN）：cPINN 的推广，各子域可使用不同网络架构和激活函数。Jagtap & Karniadakis, *Commun. Comput. Phys.* 2020。
- **B-PINN**（Bayesian PINN）：用变分推断或 HMC 估计参数后验分布，输出不确定性。Yang et al., *J. Comput. Phys.* 2021。
- **PINNsFormer**：用 Transformer 替代 MLP 作为 PINN 骨干，将 PDE 求解转化为 Seq2Seq 问题。Zhao et al., *arXiv* 2023。
- **hp-VPINN**：变分形式 PINN + hp 自适应细化，收敛性和稳定性更好。Kharazmi et al., *Comput. Methods Appl. Mech. Eng.* 2021。
- **gPINN**（Gradient-enhanced PINN）：不仅约束 PDE 残差为零，还约束残差梯度为零。Yu et al., *Comput. Methods Appl. Mech. Eng.* 2022。

### 二、傅里叶神经算子（FNO）及其变体

FNO 在频域学习 PDE 解的算子映射 $\mathcal{G}: a \mapsto u$，其中 $a$ 为输入参数场（如渗透率场），$u$ 为输出解场（如压力场）：

$$(\mathcal{K}(\phi) v_t)(x) = \mathcal{F}^{-1}(R_\phi \cdot \mathcal{F}(v_t))(x)$$

- **原始 FNO**：FFT → 频域线性变换 → IFFT，分辨率无关。Li et al., *NeurIPS* 2020。
- **U-NO**（U-shaped Neural Operator）：U-Net 架构 + FNO 谱卷积层的多尺度级联。Rahman et al., *TMLR* 2023。
- **WNO**（Wavelet Neural Operator）：用小波变换替代傅里叶变换，捕获局部奇异性和激波。Tripura & Chakraborty, *Comput. Methods Appl. Mech. Eng.* 2023。
- **LNO**（Laplace Neural Operator）：拉普拉斯变换替代傅里叶变换，对瞬态问题有更好性质。Cao et al., *Nat. Mach. Intell.* 2024。
- **球形 FNO（SFNO）**：球形调和变换（SHT）替代 Cartesian FFT，用于天气预报。Bonev et al., *ICML* 2023。

### 三、DeepONet 及其变体

深度算子网络（DeepONet）将算子学习分解为 Branch net 和 Trunk net：

- **原始 DeepONet**：Branch net 编码输入函数 → Trunk net 编码查询位置 → 内积输出。Lu et al., *Nat. Mach. Intell.* 2021。
- **MIONet**（Multiple Input Operator Network）：多输入函数版本，各自独立的 Branch net → 张量积合并。Jin et al., *SIAM J. Sci. Comput.* 2022。
- **Hyper-DeepONet**：HyperNetwork 根据 PDE 参数生成 DeepONet 权重，快速适配新 PDE 参数。de Avila Belbute-Peres et al., *arXiv* 2021。

### 四、Koopman 神经算子（KNO）

基于 Koopman 算子理论——将非线性动力系统提升到高维线性空间进行演化：

- 自编码器学习 lifting 和 inverse mapping，隐空间中线性推进时间
- 特别适合长时间演化预测
- **代表工作**：Xiong et al., *J. Comput. Phys.* 2024。
- **开源实现**：https://github.com/wtxiong/KNO

### 五、基于网格/图的 PDE 求解器

#### MeshGraphNet

在非结构网格上直接运行的图神经网络 PDE 求解器：

- 网格节点 → 图节点，网格边 → 图边，消息传递学习局部物理相互作用
- **代表工作**：Pfaff et al., *ICLR* 2021。
- **开源实现**：https://github.com/deepmind/meshgraphnets

#### GCN for PDE

经典图卷积网络应用于 PDE 网格：

- 谱域图卷积公式 $H' = \sigma(\hat{D}^{-1/2}\hat{A} \hat{D}^{-1/2} H W)$
- 简单高效，适合扩散型和椭圆型 PDE
- **代表工作**：Kipf & Welling, *ICLR* 2017; Sanchez-Gonzalez et al., *ICML* 2020。

#### GAT/GATv2 for PDE

在图卷积中引入注意力机制——动态学习边的重要性权重：

- 适合具有异质性物理现象的区域（湍流边界层、激波前后）
- **代表工作**：Veličković et al., *ICLR* 2018; Brody et al., *ICLR* 2022。

#### 等变 GNN for PDE

强制网络输出满足物理对称性（旋转等变性、平移等变性）：

- 通过球谐函数和 Clebsch-Gordan 张量积构造等变消息
- 代表性工作：Tensor Field Networks（Thomas et al., *NeurIPS* 2018）、NequIP（Batzner et al., *Nat. Commun.* 2022）

### 六、扩散模型求解 PDE

- **DiffusionPDE / DDM-PDE**：将 PDE 求解重新表述为扩散模型去噪过程，天然支持概率输出和多模态解。Gu et al., *arXiv* 2024; Huang et al., *arXiv* 2024。
- **PINN-based Diffusion Solver**：扩散模型 + PINN 物理约束引导采样，物理一致的解生成。Chen et al., *arXiv* 2024。

### 七、可微分 PDE 层 / 降阶模型

- **可微分 PDE 层**：将经典 PDE 离散格式封装为神经网络中的可微分层。可微分的有限差分算子、有限体积通量项。
- **降阶模型（ROM）**：POD（Proper Orthogonal Decomposition）对快照矩阵做主成分分析，POD-RNN / POD-Transformer 在降维模态系数上建模时间演化。
- **Autoencoder-LSTM**：用自编码器降维 + LSTM 预测隐空间时序。

### 八、AutoML for PDE / 框架与基准

- **Auto-PINN / NAS-PINN**：神经架构搜索自动设计最优 PINN 网络架构。Peddinti et al., *arXiv* 2023。
- **NVIDIA Modulus**：AI for Physics 框架，支持 PINN/FNO/DeepONet/MeshGraphNet 等。Hennigh et al., *ICCS* 2021。
- **NeuralOperators.jl**：Julia 生态中的神经算子库，与 DifferentialEquations.jl 无缝对接。Rackauckas et al., *arXiv* 2020。
- **DeepXDE / Phi-ML**：PINN 和 DeepONet 最广泛使用的教育/研究框架。Lu et al., *SIAM Rev.* 2021。
- **PDEBench**：首个系统性 AI for PDE 基准测试套件，涵盖 11 种 PDE。Takamoto et al., *NeurIPS* 2022。

## 典型化工应用

| 应用场景 | 推荐方法 | 说明 |
|----------|----------|------|
| 反应器内流场快速预测 | FNO / DeepONet | 给定入口条件和几何参数，实时输出速度场和浓度场 |
| 换热器热分布建模 | PINN / MeshGraphNet | 复杂管壳结构中的温度场预测 |
| 催化反应-扩散耦合 | PINN / cPINN | 存在化学反应源的扩散-反应方程求解 |
| 多相流流型识别与模拟 | FNO / WNO / ROM | 气-液两相流的界面追踪与压降预测 |
| 聚合反应过程动态建模 | 可微分 PDE 层 + RNN | 结合聚合动力学方程与数据驱动修正 |
| 天气预报/大气扩散 | 球形 FNO / KNO | 大规模气象场预测与化工排放扩散模拟 |

## 综合对比矩阵

| 模型 | 类型 | 计算核心 | 已迁移 | 迁移难度 | 推荐优先级 |
|------|------|----------|--------|----------|------------|
| **PINN** | PINN 族 | FC + AutoDiff | ✅ | N/A | P0（已完成） |
| **FNO** | 神经算子 | FFT + 谱乘积 | ✅ | N/A | P0（已完成） |
| **DeepONet** | 神经算子 | 双网络 + 内积 | ✅ | N/A | P0（已完成） |
| **MeshGraphNet** | 图网络 | 消息传递 | ✅ | N/A | P0（已完成） |
| **cPINN/XPINN** | PINN 变体 | 复用 PINN | 否 | 低 | P1（短期） |
| **MIONet** | 神经算子 | 复用 DeepONet | 否 | 低 | P1（短期） |
| **PINNsFormer** | PINN 变体 | Attention + MLP | 否 | 中 | P1（短期） |
| **gPINN** | PINN 变体 | 二阶 AutoDiff | 否 | 中 | P2（中期） |
| **U-NO** | 神经算子 | 复用 FNO + 采样 | 否 | 中 | P2（中期） |
| **WNO** | 神经算子 | DWT 替换 DFT | 否 | 中 | P2（中期） |
| **KNO** | 神经算子 | MLP + 线性层 | 否 | 中 | P2（中期） |
| **B-PINN** | PINN 变体 | 复用 PINN + 采样 | 否 | 中高 | P2（中期） |
| **POD-RNN** | 降阶模型 | SVD + RNN | 否 | 中高 | P2（Phase 4 已规划） |
| **LNO** | 神经算子 | Laplace 变换 | 否 | 中高 | P3（长期） |
| **球形 FNO** | 神经算子 | SHT | 否 | 高 | P3（长期） |
| **等变 GNN** | 图网络 | 球谐 + CG 积 | 否 | 高 | P3（长期） |
| **Diffusion-PDE** | 扩散模型 | U-Net + 多步采样 | 否 | 高 | P3（长期） |

## 标杆工作与资源

| 方法 | 代表性论文 | 开源实现 |
|------|-----------|----------|
| PINN | Raissi et al., *J. Comput. Phys.* 2019 | https://github.com/maziarraissi/PINNs |
| FNO | Li et al., *NeurIPS* 2020 | https://github.com/neuraloperator/neuraloperator |
| DeepONet | Lu et al., *Nat. Mach. Intell.* 2021 | https://github.com/lululxvi/deeponet |
| MeshGraphNet | Pfaff et al., *ICLR* 2021 | https://github.com/deepmind/meshgraphnets |
| KNO | Xiong et al., *J. Comput. Phys.* 2024 | https://github.com/wtxiong/KNO |
| PDEBench | Takamoto et al., *NeurIPS* 2022 | https://github.com/pdebench/PDEBench |
| Modulus | Hennigh et al., *ICCS* 2021 | https://github.com/NVIDIA/modulus |
| DeepXDE | Lu et al., *SIAM Rev.* 2021 | https://github.com/lululxvi/deepxde |

## 本仓库中的定位

本方向规划中的算子/模型包括：

- PINN 推理算子（全连接前向 + 自动微分）✅ 已发布
- FNO 推理算子（FFT → 频域变换 → IFFT 完整链路）✅ 已发布
- DeepONet 推理算子（Branch net × Trunk net 内积）✅ 已发布
- MeshGraphNet 推理算子（非结构网格上的消息传递）✅ 已发布
- PINN 变体推理算子（cPINN / XPINN / B-PINN / PINNsFormer / gPINN）
- 神经算子变体推理算子（MIONet / U-NO / WNO / KNO / LNO）
- 可微分 PDE 层算子（有限差分/有限体积封装为可微分层）
- CFD 降阶模型算子（POD-RNN / Autoencoder-LSTM）
- 扩散模型 PDE 求解算子（DiffusionPDE）

> 详细规划与待迁移模型清单请参见 [roadmap.md](../../roadmap.md) 和 [todo-algo-model.md](../../todo-algo-model.md)。