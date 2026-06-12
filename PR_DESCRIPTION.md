## 类型
- [x] 新功能
- [x] 代码重构
- [x] 文档内容更新
- [x] 构建过程或辅助工具的变动

## 概述

本 PR 是 `mat-chem-sim-pred` 仓库的一次重大架构升级，将仓库从一个面向化工行业的算子库，升级为**覆盖材料、化工、钢铁、油气等工业研发与生产业务的模拟仿真类和预测优化类全链路平台**。共提交 10 个 commit，229 个文件变更，+11,793 / -236 行。

---

## 一、仓库架构重构

### 1.1 顶层目录重组（commit `f262a90`）

将原有平铺结构重构为两大方向：

```
mat-chem-sim-pred/
├── simulation/                        # 科学计算方向
│   ├── MaterialPropertyPrediction/    #   材料性质预测与结构生成
│   ├── AI4MD/                         #   机器学习分子动力学
│   └── AI4PDE/                        #   AI for PDE
├── prediction/                        # 预测优化方向
│   ├── TabularData/                   #   表格类数据预测预训练模型
│   ├── TimeSeries/                    #   时序类数据预测预训练模型
│   └── SmallData/                     #   小数据预测优化模型
├── template/                          #   算子贡献模板
├── tutorial/                          #   教学 Notebook
├── roadmap.md                         #   开发路线图
└── FAQ.md                             #   常见问题
```

### 1.2 路线图与子领域文档（commit `8fd303f`）

- **roadmap.md**: 9 次迭代的开发路线图，覆盖 6 个子领域的 100+ 个算子开发计划
- **template/**: 4 个标准模板文件 — `algorithm.md`（算法说明）、`references.md`（参考文献）、`operator_example.md`（算子代码框架）、`test_architecture.md`（测试架构），为后续社区贡献提供统一规范
- **6 个子领域 README**: 每个子目录下均添加了 README.md，包含领域背景、核心模型/算法、CANN 实现要点和应用场景

---

## 二、科学计算方向 — Ascend C 算子实现

### 2.1 机器学习分子动力学（AI4MD）✅

已有完整的 Ascend C 算子实现（继承自 master 原有代码，本 PR 中进行了文件移动重组）：

| 算子 | 目录 | 功能 |
|------|------|------|
| **Lennard-Jones** | `simulation/AI4MD/Lennard_Jones/` | LJ 12-6 势能力场计算，包含 Host/Kernel/UT/benchmark |
| **GAFF2** | `simulation/AI4MD/GAFF2/` | GAFF2 通用力场，支持键/角/二面角/非键相互作用 |
| **PME** | `simulation/AI4MD/PME/` | Particle Mesh Ewald 长程静电计算，含 Ewald 实空间/倒空间/FFT 加速 |
| **SHAKE** | `simulation/AI4MD/SHAKE/` | SHAKE 键长约束算法 |
| **Velocity Verlet** | `simulation/AI4MD/velocity-verlet/` | Velocity Verlet 时间积分器 + 恒温器 |
| **DPD** | `simulation/AI4MD/Dissipative_particle_dynamics/` | 耗散粒子动力学 |

### 2.2 AI for PDE（commit `d6410a4`）✅

新增 4 个 AI4PDE 子方向的完整 Ascend C 算子实现：

| 算子 | 目录 | 内容 |
|------|------|------|
| **PINN** | `simulation/AI4PDE/pinn/` | 物理信息神经网络推理算子（Host + Kernel + UT） |
| **FNO** | `simulation/AI4PDE/fno/` | 傅里叶神经算子（频谱卷积 + 非线性变换） |
| **DeepONet** | `simulation/AI4PDE/deeponet/` | 深度算子网络（Branch Net + Trunk Net） |
| **MeshGraphNet** | `simulation/AI4PDE/mesh_graph_net/` | 网格图网络（图卷积 + 消息传递） |
| **pde_common** | `simulation/AI4PDE/pde_common/` | PDE 公共工具模块 |

### 2.3 材料性质预测（commit `d6410a4`）

- **DAO PyTorch 算子库** (`simulation/MaterialPropertyPrediction/DAO/`): 基础材料科学算子参考库，包含原子描述符计算、图构建、特征变换等核心模块

---

## 三、预测优化方向 — PyTorch 参考实现

### 3.1 表格类数据预测（commit `8d96f42`）🔧

| 模型 | 文件 | 内容 |
|------|------|------|
| **TabNet** | `prediction/TabularData/tabnet.py` (261 行) | SparseMax / GLU Block / FeatureTransformer / AttentiveTransformer / Encoder / Classifier / Regressor |
| 测试 | `prediction/TabularData/test_tabnet.py` (146 行) | 合成数据分类 + 回归 + 特征重要性测试 |

### 3.2 时序类数据预测（commit `8d96f42`）🔧

| 模型 | 文件 | 内容 |
|------|------|------|
| **TimesNet** | `prediction/TimeSeries/timesnet.py` (233 行) | FFT 周期发现 / InceptionBlock / TimesBlock / 多周期折叠卷积 |
| 测试 | `prediction/TimeSeries/test_timesnet.py` (149 行) | 单变量 + 多变量时序预测测试 |

### 3.3 小数据预测优化（commit `8d96f42`）🔧

| 模型 | 文件 | 内容 |
|------|------|------|
| **高斯过程回归** | `prediction/SmallData/gpr.py` (317 行) | RBF/Matern/RQ 核函数 / GP 回归 / 边际似然 / EI/UBC/PI 采集函数 / 贝叶斯优化器 / NumPy 兼容接口 |
| 核函数 | `prediction/SmallData/kernels.py` (187 行) | RBF / Matern / RationalQuadratic / Periodic / Linear / WhiteNoise 及组合核 |
| 测试 | `prediction/SmallData/test_gpr.py` (218 行) | GP 回归拟合 + 预测 + 贝叶斯优化完整流程 |

### 3.4 Python 包结构修复（commit `7d239a6`）

- 为 `prediction/`、`TabularData/`、`TimeSeries/`、`SmallData/` 创建 `__init__.py` 文件
- 修复 EI/PI 采集函数中的设备不匹配问题（`torch.tensor` 未指定 device/cuda）
- 修复 `test_gpr.py` 函数内局部导入问题
- 添加 `log_marginal_likelihood()` 未 fit 的友好错误处理

---

## 四、教学与入门体系（commit `ef3d7d2`）

| Notebook | 文件 | 主题 |
|----------|------|------|
| 00 | `tutorial/00_Environment_Setup.ipynb` | 环境准备（Python/PyTorch/CANN 检查） |
| 01 | `tutorial/01_AscendC_Basics.ipynb` | Ascend C 基础（算子结构/开发流程/构建） |
| 02 | `tutorial/02_MaterialPrediction.ipynb` | 材料性质预测（晶体 GNN/DAO 算子库） |
| 03 | `tutorial/03_AI4MD_Tutorial.ipynb` | 机器学习分子动力学（LJ 力场/MD 流程） |
| 04 | `tutorial/04_AI4PDE_Tutorial.ipynb` | AI for PDE（PINN/FNO/DeepONet/MeshGraphNet） |
| 05 | `tutorial/05_TabularData.ipynb` | 表格数据预测（TabNet/特征选择） |
| 06 | `tutorial/06_TimeSeries.ipynb` | 时序数据预测（TimesNet/周期发现） |
| 07 | `tutorial/07_SmallData.ipynb` | 小数据预测（GP 回归/贝叶斯优化） |

共 8 个 Jupyter Notebook，覆盖环境搭建、基础教学和 6 个子领域的算子和模型使用。

---

## 五、基础设施与 DevOps（commit `4f43db9`）

### 5.1 CI/CD 流水线

| 文件 | 用途 |
|------|------|
| `.gitcode/workflows/ci.yml` | 代码风格检查 + PyTorch 模型测试（Lint + pytorch-tests） |
| `.gitcode/workflows/operator-build.yml` | Ascend C 算子构建与测试（7 个 Job：LJ/GAFF2/PME/SHAKE/VV/PINN/FNO/DeepONet/MeshGraphNet/DPD），使用 YAML anchor 消除冗余配置 |
| `.gitcode/workflows/nightly.yml` | 夜间构建（全量算子构建 + 性能基准测试） |

### 5.2 开发容器

| 文件 | 用途 |
|------|------|
| `.devcontainer/Dockerfile` | 基于 Ubuntu 22.04，预装 cmake/gcc/Python/PyTorch/CANN 依赖 |
| `.devcontainer/devcontainer.json` | VS Code DevContainer 配置（环境变量/扩展/挂载） |
| `.devcontainer/setup.sh` | 容器启动后自动配置脚本（环境验证/依赖安装/hooks） |

### 5.3 仓库配置

| 文件 | 用途 |
|------|------|
| `.gitignore` | 排除 `Reference/`（外部参考仓库）、`__pycache__/`、`*.pyc` |
| `AGENTS.md` | AI 开发辅助规则 |

---

## 六、文档体系

### 6.1 FAQ（commit `52852c3` → `f2bac1c`）

`FAQ.md` 涵盖：
- 仓库定位说明（Accend C + PyTorch 双轨制）
- SIMT vs SIMD 架构差异对比表
- FP64 精度支持情况与替代方案
- Ascend 910B/910C/950 系列芯片支持矩阵
- PyTorch 代码如何在 Ascend 上运行
- 如何选择 Ascend C 直接开发 vs pypto 迁移

### 6.2 README 更新（多次迭代）

- 重新定义仓库定位：覆盖材料、化工、钢铁、油气等工业领域
- 添加 6 个重点关注领域表格（含状态/目标/关键算子）
- 添加贡献模板说明
- 添加 18 个 CANN 官方参考仓库章节（按基础数学库/开发标准与调试/AI Kernel 生成/相关参考库分类）
- 添加 FAQ 跳转链接

---

## 如何测试

### 1. 预测优化 PyTorch 模型测试
```bash
cd prediction
python -m pytest TabularData/test_tabnet.py -v
python -m pytest TimeSeries/test_timesnet.py -v
python -m pytest SmallData/test_gpr.py -v
```

### 2. CI 流水线（需 CANN 环境）
```bash
# CI: Lint + PyTorch 测试
# 自动化在 .gitcode/workflows/ci.yml 中定义

# 算子构建
# 自动化在 .gitcode/workflows/operator-build.yml 中定义
```

### 3. Tutorial 使用
```bash
cd tutorial
jupyter notebook 00_Environment_Setup.ipynb
```

---

## Checklist
- [x] 我的代码遵循这个项目的代码风格
- [x] 我已经自己测试过我的代码
- [x] 我已经更新了相应的文档
- [x] 我已经根据需要更新了对应的变更日志
- [x] 我已经在标题中正确使用了类型标签

## 其他信息

### 变更统计
- **229 个文件变更**
- **+11,793 行插入** / **-236 行删除**
- **10 个 commit**

### 后续计划
1. 预测优化方向（TabularData/TimeSeries/SmallData）从 PyTorch 参考实现逐步迁移为 Ascend C 高性能算子
2. 材料性质预测方向补充晶体图神经网络的 Ascend C 实现
3. 完善各算子的精度对比和性能基准数据

### Reviewer 关注点
- `simulation/AI4MD/` 下的原有算子文件仅进行了目录移动重组，算子逻辑无变更
- `prediction/` 下的 PyTorch 实现为参考代码，后续将作为 Ascend C 迁移的 base
- CI 配置中的 YAML anchor 模式请确认与 GitCode Actions runner 兼容