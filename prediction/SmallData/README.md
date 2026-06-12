# 小数据预测优化模型（Small Data Prediction & Optimization）

## 领域简介

化工行业面临一个普遍挑战：**标记数据稀缺**。原因包括：

- **实验成本高昂**：一次催化活性测试、力学实验或反应条件筛选需要数小时到数天
- **长周期测试**：稳定性测试、老化实验、甚至数月的批次反应
- **安全与法规限制**：涉及危险品或受监管物质的实验受限
- **数据私密性**：企业不愿共享核心工艺数据

因此，化工 AI 场景中常见的情况是只有几百甚至几十个标记样本。在此类"小数据"环境下的预测与优化，无法直接套用大数据深度学习范式，需要专门的方法：高斯过程回归、贝叶斯神经网络、主动学习、小样本学习、迁移学习等。这些方法的核心特点是：**显式建模不确定性、利用先验知识、高效利用有限样本**。

本方向聚焦化工行业的小数据场景，为配方优化、工艺参数搜索、新催化剂开发等提供从数据采集到模型部署的完整方案。

## 关键模型与算法

### 一、高斯过程回归（Gaussian Process Regression）

GP 是小数据回归的经典方法：

- **核函数**：RBF（径向基函数）、Matern（控制平滑度）、周期核（化学周期性）、线性核（趋势）等
- **预测输出**：均值 $\mu(x)$ + 方差 $\sigma^2(x)$（不确定性量化）
- **训练**：通过最大化对数边际似然优化核超参数
- **推理**：$O(n^3)$ 的 Cholesky 分解（$n$ 为训练样本数），对大规模数据需要稀疏近似
- **稀疏 GP（SVGP）**：引入 $m$ 个诱导点，变分推断将复杂度降至 $O(nm^2)$。Hensman et al., *UAI* 2013。
- **随机傅里叶特征 GP（RFF-GP）**：通过 Bochner 定理将核函数近似为正弦/余弦基函数，复杂度降至 $O(n D^2)$。Rahimi & Recht, *NeurIPS* 2007。
- **当前状态**：已实现 PyTorch 参考代码 [gpr.py](gpr.py) + [kernels.py](kernels.py)。

### 二、贝叶斯神经网络（Bayesian Neural Network, BNN）

将贝叶斯推断引入神经网络权重的学习：

- **变分推断 BNN（Bayes by Backprop）**：权重视为随机变量，假设高斯近似后验，最小化 KL 散度。Blundell et al., *ICML* 2015。
- **MC Dropout**：在推理时保持 Dropout 开启，多次前向采样近似后验分布。Gal & Ghahramani, *ICML* 2016。
- **Laplace 近似（Laplace Redux）**：在 MAP 解附近用高斯分布近似后验，使用 GGN 近似 Hessian，事后添加不确定性量化。Daxberger et al., *NeurIPS* 2021。
- **SWAG**：收集 SGD 轨迹上的权重快照，构建高斯近似后验。Maddox et al., *NeurIPS* 2019。
- **SNGP**：谱归一化 + 随机傅里叶特征层，一个网络即可获得类似 Deep Ensemble 的不确定性质量。Liu et al., *NeurIPS* 2020。
- **Deep Ensembles**：M 个独立初始化的相同架构网络，取均值和方差作为预测。Lakshminarayanan et al., *NeurIPS* 2017。

### 三、多保真度高斯过程（Multi-fidelity GP）

工业中常见的"低成本仿真+昂贵实验"数据融合场景：

- **NARGP**（Nonlinear Auto-Regressive GP）：将线性自回归推广为非线性映射 $g(f_{t-1}(x), x)$，低保真输出作为完整输入特征。Perdikaris et al., *Proc. R. Soc. A* 2017。
- **ResGP / MF-DGP**（Multi-fidelity Deep GP）：用深度 GP 层级建模残差映射，底层 GP 拟合低保真趋势，上层 GP 建模高保真偏差。Cutajar et al., *arXiv* 2019。
- **线性自回归 co-kriging**：经典的 Kennedy-O'Hagan 方法，$f_{high}(x) = \rho f_{low}(x) + \delta(x)$。Kennedy & O'Hagan, *Biometrika* 2000。

### 四、贝叶斯优化（Bayesian Optimization, BO）

全局黑箱优化的标准方法，尤其适合实验/仿真代价高昂的情况：

- **代理模型**：GP / RF / BNN 作为代理模型，提供预测均值和不确定性
- **采集函数**：EI（Expected Improvement）、PI（Probability of Improvement）、UCB（Upper Confidence Bound）
- **TuRBO**（Trust Region BO）：针对高维黑箱优化（d > 20），维护多个局部信赖域，突破维度诅咒。Eriksson et al., *NeurIPS* 2019。
- **SAASBO**：通过 Horseshoe 先验在 GP 核上引入稀疏性，自动发现高维空间中的重要维度。Eriksson & Jankowiak, *UAI* 2021。
- **HEBO**：华为诺亚方舟实验室的 BO 框架——异方差 GP + 输入/输出空间变换 + 多目标采集。Cowen-Rivers et al., *JAIR* 2022。NeurIPS 2020 黑箱优化竞赛获胜。
- **SMAC3**：以随机森林为代理模型的 BO，天然支持分类/条件参数空间。Lindauer et al., *JMLR* 2022。
- **Hyperopt (TPE)**：Tree-structured Parzen Estimator，非参数密度估计取代 GP。Bergstra et al., *NeurIPS* 2011。
- **Optuna**：日本 Preferred Networks 的 BO 框架，定义-运行风格 API，灵活度极高。Akiba et al., *KDD* 2019。
- **BOHAMIANN**：BNN + HMC 采样的 BO 代理模型。Springenberg et al., *NeurIPS* 2016。
- **当前状态**：已实现基本 BO（EI/PI/UCB）参考代码 [gpr.py](gpr.py)。

### 五、小样本迁移学习（Few-shot Transfer Learning）

从源域（公开数据集+模拟数据）预训练到目标域（少量化工实验数据）微调：

- **MAML**（Model-Agnostic Meta-Learning）：跨任务学习良好的初始化参数，新任务仅需少量梯度步即可适配。Finn et al., *ICML* 2017。
- **Reptile**：MAML 的简化版，完全一阶梯度，无二阶依赖。Nichol et al., *arXiv* 2018。
- **Prototypical Networks**：每个类原型 = 支撑集样本嵌入的均值，查询点分类到最近原型。Snell et al., *NeurIPS* 2017。
- **参数高效微调**：Adapter、LoRA、Prefix-tuning 在化工场景下的适配。
- **域适应**：最大均值差异（MMD）或对抗训练缩小源域和目标域的分布差异。
- **OT-based Transfer (JDOT / DeepJDOT)**：利用最优传输理论最小化源域和目标域的联合分布差异。Courty et al., *NeurIPS* 2017; Damodaran et al., *ECCV* 2018。

### 六、主动学习（Active Learning）

在标记预算有限时，主动选择最有标注价值的数据：

- **不确定性采样**：熵（Entropy）、边际（Margin）、BALD（Bayesian Active Learning by Disagreement）。Houlsby et al., *arXiv* 2011。
- **BatchBALD**：批量选择，通过联合互信息避免冗余采样。Kirsch et al., *NeurIPS* 2019。
- **多样性采样**：CoreSet（K-Center Greedy，几何代表性）、BADGE（梯度嵌入 + K-Means++ 多样性）。Sener & Savarese, *ICLR* 2018; Ash et al., *ICLR* 2020。

### 七、深度核学习（Deep Kernel Learning, DKL）

将深度神经网络作为特征提取器嵌入到 GP 核中：

- $k_{DKL}(x, x') = k_{GP}(g_\phi(x), g_\phi(x'))$，DNN 提取非线性特征，GP 在特征空间中进行概率回归
- 兼具 DNN 的表达能力和 GP 的不确定性量化
- **代表工作**：Wilson et al., *AISTATS* 2016。

### 八、物理信息高斯过程（Physics-Informed GP, PIGP）

将物理定律（PDE 微分约束）作为 GP 的线性算子约束：

- 若 $f \sim GP(0,k)$，则 $L_x f \sim GP(0, L_x L_y k)$（$L$ 为线性微分算子）
- 通过协方差函数的微分将物理方程融入 GP 先验
- 在极少量实验数据下实现高精度插值和外推
- **代表工作**：Raissi et al., *J. Comput. Phys.* 2017; Raissi & Karniadakis, *J. Comput. Phys.* 2018。

### 九、数据增强（Data Augmentation）

从少量真实样本生成合理合成样本：

- **VAE 增强**：在潜在空间中采样新样本，解码回原始特征空间
- **GAN 增强**：条件 GAN（cGAN）以类别或属性为条件生成新样本
- **扩散模型增强**：生成质量高、模式覆盖广
- **化工特定增强**：反应条件扰动（温度 ±5K、压力 ±0.1MPa）、配方配比插值

### 十、理论保证方法

- **PAC-Bayes 泛化界**：为随机化分类器提供可证明的泛化保证，在极少样本下给出理论保证。Dziugaite & Roy, *ICML* 2017; Pérez-Ortiz et al., *JMLR* 2021。

## 典型化工应用

| 应用场景 | 推荐方法 | 说明 |
|----------|----------|------|
| 催化剂配方的小样本优化 | 贝叶斯优化 + GP | 仅 20-50 次实验找到最优催化剂配比 |
| 聚合物合成条件搜索 | HEBO / TuRBO | 高维参数空间的高效优化 |
| 设备故障类型识别 | Prototypical Network | 每类故障仅 1-5 个样本的极端小样本分类 |
| 新反应路径筛选 | GP 分类器 + 迁移学习 | 从已知反应迁移先验知识，预测新反应的可行性 |
| 实验设计（DoE）增强 | 主动学习采样 | 在高维工艺参数空间中选择最有效的实验点 |
| 模拟+实验数据融合 | NARGP / MF-DGP | 低成本仿真数据 + 少量昂贵实验数据的联合建模 |
| 物理增强小样本预测 | PIGP / DKL | 结合物理方程先验的高精度插值外推 |

## 综合对比矩阵

| 序号 | 模型 | 理论成熟度 | 小样本效果 | Ascend 迁移难度 | 工业落地优先级 | 当前仓库状态 |
|------|------|:----------:|:----------:|:---------------:|:-------------:|-------------|
| 1 | **标准 GP** | 5/5 | 5/5 | 低 | **P0** | ✅ 已实现 ([gpr.py](gpr.py)) |
| 2 | **HEBO** | 5/5 | 5/5 | 低 | **P0** | 华为自研，极高优先级 |
| 3 | **MC Dropout** | 4/5 | 3/5 | 极低 | **P0** | roadmap 中 |
| 4 | **Deep Ensembles** | 5/5 | 4/5 | 极低 | **P0** | 通用 DNN 可覆盖 |
| 5 | **TuRBO (高维BO)** | 4/5 | 5/5 | 低 | **P1** | roadmap BO 可扩展 |
| 6 | **NARGP (多保真GP)** | 4/5 | 4/5 | 中等 | **P1** | 工业 MF 场景刚需 |
| 7 | **PIGP (物理信息GP)** | 3/5 | 5/5 | 中等 | **P1** | 与 AI4MD 链路协同 |
| 8 | **SNGP** | 4/5 | 4/5 | 低 | **P1** | 未规划 |
| 9 | **Laplace Approximation** | 4/5 | 4/5 | 极低 | **P1** | 未规划 |
| 10 | **Prototypical Networks** | 5/5 | 3/5 | 低 | **P2** | roadmap 中 |
| 11 | **DKL** | 4/5 | 4/5 | 中等 | **P2** | 未规划 |
| 12 | **MAML / Reptile** | 4/5 | 3/5 | 低 | **P2** | 未规划 |
| 13 | **SVGP (稀疏GP)** | 5/5 | 4/5 | 中等 | **P2** | 未规划 |
| 14 | **OT Transfer (JDOT)** | 3/5 | 3/5 | 中等 | **P3** | 未规划 |

## 相关资源

- **GPyTorch**（https://github.com/cornellius-gp/gpytorch）：高斯过程高效工具库
- **BoTorch**（https://github.com/pytorch/botorch）：贝叶斯优化库
- **scikit-optimize**（https://github.com/scikit-optimize/scikit-optimize）：轻量 BO 工具
- **learn2learn**（https://github.com/learnables/learn2learn）：小样本学习框架
- **Open Catalyst Project**（https://opencatalystproject.org/）：催化领域公开基准
- **HEBO**（https://github.com/huawei-noah/HEBO）：华为诺亚方舟 BO 框架

## 本仓库中的定位

本方向规划中的算子/模型包括：

- 高斯过程回归算子（核矩阵构建 + Cholesky 分解/共轭梯度）✅ PyTorch 参考实现
- 贝叶斯优化算子（EI / PI / UCB 采集函数 + 多目标支持）✅ PyTorch 参考实现
- 多保真度 GP 算子（NARGP / MF-DGP）
- 高维贝叶斯优化算子（HEBO / TuRBO / SAASBO）
- 贝叶斯神经网络算子（MC Dropout / Laplace 近似 / SNGP / SWAG）
- 主动学习采样算子（不确定性 / 多样性采样 / BatchBALD）
- 小样本迁移学习算子（MAML / Reptile / Prototypical Network）
- 深度核学习算子（DKL：DNN 特征提取 + GP 推理）
- 物理信息 GP 算子（PIGP：微分约束 + GP 先验）
- 数据增强算子（VAE / GAN / Diffusion 生成）

> 详细规划与待迁移模型清单请参见 [roadmap.md](../../roadmap.md) 和 [todo-algo-model.md](../../todo-algo-model.md)。