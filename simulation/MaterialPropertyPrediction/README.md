# 材料性质预测与材料结构生成（Material Property Prediction & Structure Generation）

## 领域简介

材料性质预测与结构生成是计算材料科学的核心课题，旨在通过数据驱动的方法加速新材料发现与性能评估。传统方法依赖第一性原理计算（DFT）或经典力场模拟，计算成本高、通量低；近年来基于机器学习的材料模型在保持预测精度的同时，将推理速度提升数个数量级。

本方向承接 "**模拟 → 数据 → 预测 → 设计 → 优化**" 链路中的数据产出与预测环节：上游分子动力学/DFT 模拟产出结构-性质数据，本方向则利用这些数据构建材料性质预测模型和逆向结构生成模型，支撑催化剂筛选、电池材料设计、聚合物开发等化工场景。

## 关键模型与算法

### 一、晶体图神经网络（Crystal GNN）

以晶体结构为输入的图神经网络，原子为节点、键为边，在网络的消息传递过程中学习局部化学环境与长程有序性的联合表征：

- **CGCNN**（Crystal Graph Convolutional Neural Network）：最早将图卷积引入晶体性质预测的模型，预测带隙、形成能、体积模量等目标性质。Xie & Grossman, *Phys. Rev. Lett.* 2018。
- **MEGNet**（MatErials Graph Network）：引入多图（原子图、键图、全局状态图）架构，支持多任务学习。Chen et al., *J. Am. Chem. Soc.* 2019。
- **GATGNN**（Global Attention Transformer GNN）：在消息传递中引入注意力机制，区分不同化学环境对目标性质的贡献。Louis et al., *npj Comput. Mater.* 2020。
- **MatGL**（Materials Graph Library）：Materials Virtual Lab 推出的统一 PyTorch Geometric 材料 GNN 库，对 M3GNet、MEGNet、CHGNet 等做 PyTorch 标准化重写。Ko et al., 2023-2024。
- **CHGNet**（Crystal Hamiltonian Graph Network）：在 Materials Project 150 万+无机晶体结构上预训练的通用 GNN，显式建模电荷（磁矩）信息。Deng et al., *Nat. Mach. Intell.* 2023。
- **M3GNet**（Materials 3-body Graph Network）：引入三体相互作用（键角信息）的通用原子间势能模型，在 Materials Project 数据上训练。Chen & Ong, *Nat. Comput. Sci.* 2022。

这些模型通常在大规模公开数据集（如 Materials Project、OQMD、NOMAD）上预训练，可迁移至小样本化工材料体系。

### 二、等变图神经网络（Equivariant GNN）

利用 SE(3)/E(3) 等变性保证模型对旋转、平移、反演对称性的物理一致性，显著提升数据效率和精度：

- **MACE / MACE-MP-0**：多体等变消息传递统一框架，MACE-MP-0 覆盖 89 种元素，是 Materials Project 官方推荐的通用势。Batatia et al., *NeurIPS* 2022; *arXiv* 2024。
- **Equiformer / EquiformerV2**：将 Transformer 注意力与 SE(3) 等变性结合，在 OC20/OC22 催化剂任务上取得 SOTA。Liao & Smidt, *ICLR* 2023; Liao et al., *ICLR* 2024。
- **Allegro**：严格局域的等变原子间势能，无消息传递，推理速度在等变网络中领先。Musaelian et al., *Nat. Commun.* 2023。
- **SevenNet**：基于 NequIP 的大规模等变势能，在 GNoME 生成的结构数据上预训练，覆盖 10^6+ 结构。Park et al., *ACS Nano* 2024。
- **ORB**：面向大规模材料筛选优化的小型高性能模型，知识蒸馏 + 轻量等变 GNN。Neumann et al., *arXiv* 2024。

### 三、材料基础模型（Foundation Models）

- **GNoME**：DeepMind 的规模化材料发现框架，结合 GNN 预测 + 主动学习 + DFT 验证，预测 220 万稳定晶体结构。Merchant et al., *Nature* 2023。
- **MatterSim / JMP**：Microsoft Research 的材料基础模型系列，基于 Graphormer 架构，覆盖温度 0-5000K、压力 0-1000 GPa。Yang et al., *Nature* 2024; Shoghi et al., *ICLR* 2024。
- **FAIR-Chem 系列**：Meta FAIR 的 Open Catalyst 系列等变模型（GemNet-OC、SCN、eSCN），专注催化表面反应。Gasteiger et al., *TMLR* 2022; Passaro & Zitnick, *ICML* 2023。

### 四、材料描述符计算

在 ML 模型之外，基于物理启发的描述符（Descriptor）依然是高通量筛选的重要工具：

- **组分描述符**：元素比例、电负性差、平均原子半径、价电子数
- **结构描述符**：配位数、多面体扭曲度、键长分布、孔隙率
- **电子描述符**：d 带中心、Bader 电荷、态密度矩
- **DimeNet++ 基函数**：BesselBasisLayer / SphericalBasisLayer / SinusoidsEmbedding。Gasteiger et al., *ICLR* 2020; *NeurIPS* 2020 Workshop。

描述符可以独立回归性质，也可作为 GNN 输入特征或特征工程补充。

### 五、晶体结构生成模型

逆向设计——给定目标性质，生成满足条件的稳定晶体结构：

- **Diffusion 模型**：CDVAE（ICLR 2022）、DiffCSP（NeurIPS 2023）、MatterGen（Nature 2025）——在原子坐标和晶格参数上定义扩散过程，从噪声逐步去噪得到合理结构。
- **Flow 模型**：FlowMM —— 基于 Riemannian Flow Matching 的晶体生成，训练更稳定、推理步数更少。Miller et al., *ICML* 2024。
- **CrysFormer / DAO**：用孪生网络 + 晶格增强自注意力的晶体结构预测基础模型。DAO Team, *2024*。
- **CrystalFormer**：将晶体结构生成转化为序列生成问题，GPT-style 自回归预测 Wyckoff 位置和元素类型。Cao et al., *Sci. Adv.* 2024。
- **GAN/VAE 方法**：较早期的尝试，生成质量不及 Diffusion 模型，但推理速度更快。

### 六、相图预测与相变识别

- **聚类 + 热力学模型**：从 MD/DFT 数据中识别有序相和无序相，结合 CALPHAD 方法预测相图和相变温度。
- **主动学习驱动的相图构建**：在选择性子集上采样，减少 DFT 计算量，增量构建精确相边界。

### 七、聚合物性质预测

- **Polymet / GNN for polymer**：将聚合物表示为重复单元序列图，预测玻璃化温度（Tg）、介电常数、气体渗透率、结晶度等。
- **手性/立构规整性编码**：在嵌入层区分等规、间规、无规立构。

### 八、配方-结构-性能关联

化工配方设计中的多模态融合：

- **输入模态**：配方组成（表格）、工艺参数（温度、压力、时间）、微观结构图像（SEM/TEM）
- **融合策略**：以 Transformer 或交叉注意力网络为骨干，输出宏观性能（强度、硬度、催化活性）

## 典型化工应用

| 应用场景 | 关键模型 | 目标性质 |
|----------|----------|----------|
| 催化剂筛选 | CGCNN / GATGNN / EquiformerV2 | 吸附能、反应势垒、d 带中心 |
| 电池电极/电解质设计 | MEGNet / M3GNet / Diffusion 生成 | 离子电导率、带隙、形成能 |
| 高分子材料开发 | Polymer GNN | Tg、模量、气体渗透率 |
| 合金/金属玻璃 | 描述符 + GP 回归 | 强度、韧性、腐蚀电位 |
| 相变材料筛选 | 主动学习 + 相图模型 | 相变温度、潜热 |
| 高通量材料发现 | MACE-MP-0 / GNoME / MatterSim | 形成能、稳定性、带隙 |

## 相关资源

- **Materials Project**（https://next-gen.materialsproject.org/）：约 15 万无机晶体结构及性质数据
- **OQMD**（https://oqmd.org/）：约 100 万材料条目
- **NOMAD**（https://nomad-lab.eu/）：DFT 原始数据仓库
- **MatBench**（https://matbench.materialsproject.org/）：材料性质预测基准
- **MatGL**（https://github.com/materialsvirtuallab/matgl）：PyTorch 材料 GNN 统一库
- **MACE**（https://github.com/ACEsuit/mace）：多体等变势能模型
- **FAIR-Chem**（https://github.com/FAIR-Chem/fairchem）：Open Catalyst 系列模型

## 本仓库中的定位

本方向规划中的算子/模型包括：

- 材料晶体图神经网络（CGCNN / MEGNet / GATGNN / MatGL / CHGNet / M3GNet）推理算子
- 等变 GNN 推理算子（MACE / Equiformer / Allegro / SevenNet / ORB）
- 材料基础模型推理算子（GNoME / MatterSim / FAIR-Chem）
- 材料描述符批量化计算算子（组分/结构/电子描述符 + DimeNet++ 基函数）
- 晶体结构生成模型（Diffusion / Flow / CrysFormer / CrystalFormer）推理算子
- 相图预测与相变识别算子
- 聚合物性质预测算子（Polymer GNN）
- 配方-结构-性能多模态融合算子

> 详细规划与待迁移模型清单请参见 [roadmap.md](../../roadmap.md) 和 [todo-algo-model.md](../../todo-algo-model.md)。