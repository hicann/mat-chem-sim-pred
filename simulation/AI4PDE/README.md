# AI for PDE — 物理信息驱动与神经算子模型

## 领域简介

化工过程中的大量物理场模拟（流体力学、传热传质、结构力学、反应-扩散耦合）本质上是偏微分方程（PDE）的求解问题。传统方法（有限差分 FDM、有限体积 FVM、有限元 FEM）在网格离散与数值迭代上消耗大量计算资源，尤其在高维参数化、多物理场耦合和实时预测场景中存在瓶颈。

近年来，**AI for PDE**（或称科学机器学习、AI-driven simulation）兴起，核心理念是用神经网络直接学习 PDE 的解算子或作为物理约束的近似求解器，在保持一定精度的前提下将推理速度提升 1-3 个数量级。本方向涵盖物理信息神经网络（PINN）、神经算子（Neural Operator）、网格上的图神经网络、可微分 PDE 层等路线。

## 关键模型与算法

### 物理信息神经网络（PINN）

PINN 将 PDE 的残差作为损失函数的正则化项，在无标签或少量标签下训练神经网络满足物理规律：

$$
\mathcal{L} = \mathcal{L}\_{\text{data}} + \lambda \cdot \mathcal{L}\_{\text{PDE}}
$$

其中 $\mathcal{L}_{\text{PDE}}$ 包含控制方程残差、边界条件和初始条件。推理时，网络直接输出场变量（速度场 $\mathbf{u}$、压力场 $p$、温度场 $T$、浓度场 $c$）。

- **原创工作**：Raissi et al., *J. Comput. Phys.* 2019。
- **变体**：VPINN（变分 PINN）、cPINN（守恒 PINN）、XPINN（分区 PINN）、ParPINN（并行 PINN）。
- **算子化要点**：自动微分（AutoDiff）是 PINN 的核心算子需求，需在 Ascend C 上高效实现梯度计算。

### 傅里叶神经算子（FNO）

FNO 在频域学习 PDE 解的算子映射 $\mathcal{G}: a \mapsto u$，其中 $a$ 为输入参数场（如渗透率场），$u$ 为输出解场（如压力场）：

$$
(\mathcal{K}(\phi) v_t)(x) = \mathcal{F}^{-1}(R_\phi \cdot \mathcal{F}(v_t))(x)
$$

- **核心优势**：分辨率无关（discretization-invariant），训练后可一次性求解任意网格上的 PDE。
- **原创工作**：Li et al., *NeurIPS* 2020。
- **算子化要点**：FFT → 频域线性变换 → IFFT 的完整链路需算子化实现，含复数运算支持。

### DeepONet

深度算子网络（DeepONet）将算子学习分解为 Branch net 和 Trunk net：

- **Branch net**：编码输入函数 $u$ 在传感器点上的离散值
- **Trunk net**：编码查询位置 $y$
- **输出**：$\mathcal{G}(u)(y) = \sum_k b_k(u) \cdot t_k(y)$

- **原创工作**：Lu et al., *Nat. Mach. Intell.* 2021。
- **适用于**：输入函数与输出解不在同一网格的场景。

### MeshGraphNet

在非结构网格上直接运行的图神经网络 PDE 求解器：

- 网格节点 → 图节点，网格边 → 图边
- 消息传递过程学习节点间的局部物理相互作用
- 支持编码器-处理器-解码器架构，可用于稳态和瞬态问题

- **原创工作**：Pfaff et al., *ICLR* 2021。
- **适用于**：复杂几何边界上的 CFD 模拟、流固耦合。

### 可微分 PDE 层

将经典 PDE 离散格式封装为神经网络中的可微分层，实现传统方法与深度学习的混合：

- **示例**：可微分的有限差分算子、可微分的有限体积通量项
- **优势**：保留物理可解释性，同时支持端到端反向传播优化

### 降阶模型（ROM）

对高维 CFD 仿真数据进行降维后建立时序预测模型：

- **POD（Proper Orthogonal Decomposition）**：对快照矩阵做主成分分析，提取主导模态
- **POD-RNN / POD-Transformer**：在降维后的模态系数上建模时间演化
- **Autoencoder-LSTM**：用自编码器降维 + LSTM 预测隐空间时序

## 典型化工应用

| 应用场景 | 推荐方法 | 说明 |
|----------|----------|------|
| 反应器内流场快速预测 | FNO / DeepONet | 给定入口条件和几何参数，实时输出速度场和浓度场 |
| 换热器热分布建模 | PINN / MeshGraphNet | 复杂管壳结构中的温度场预测 |
| 催化反应-扩散耦合 | PINN | 存在化学反应源的扩散-反应方程求解 |
| 多相流流型识别与模拟 | FNO / ROM | 气-液两相流的界面追踪与压降预测 |
| 聚合反应过程动态建模 | 可微分 PDE 层 + RNN | 结合聚合动力学方程与数据驱动修正 |

## 标杆工作与资源

| 方法 | 代表性论文 | 开源实现 |
|------|-----------|----------|
| PINN | Raissi et al., *J. Comput. Phys.* 2019 | https://github.com/maziarraissi/PINNs |
| FNO | Li et al., *NeurIPS* 2020 | https://github.com/neuraloperator/neuraloperator |
| DeepONet | Lu et al., *Nat. Mach. Intell.* 2021 | https://github.com/lululxvi/deeponet |
| MeshGraphNet | Pfaff et al., *ICLR* 2021 | https://github.com/deepmind/meshgraphnets |

## 本仓库中的定位

本方向规划中的算子/模型包括：

- PINN 推理算子（全连接前向 + 自动微分）
- FNO 推理算子（FFT → 频域变换 → IFFT完整链路）
- DeepONet 推理算子（Branch net × Trunk net 内积）
- MeshGraphNet 推理算子（非结构网格上的消息传递）
- 可微分 PDE 层算子（有限差分/有限体积封装为可微分层）
- CFD 降阶模型算子（POD-RNN / Autoencoder-LSTM）

> 详细规划请参见 [roadmap.md](../../roadmap.md)。