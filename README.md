# mat-chem-sim-pred

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/mat-chem-sim-pred)

## 项目简介

本仓库为基于华为 CANN（Ascend C）计算框架开发的化工行业专用算子库，聚焦**科学计算**与**预测优化**两大方向。通过高性能算子加速分子模拟、材料性质预测、反应路径优化、工艺过程建模等核心任务，为化工研发、智能制造及材料创新提供高效计算能力支撑。

核心定位：构建"**模拟 → 数据 → 预测 → 设计 → 优化**"的全链路化工 AI 算子体系。

## 仓库结构

```
mat-chem-sim-pred/
├── simulation/                              # 科学计算方向
│   ├── MaterialPropertyPrediction/          # 材料性质预测与材料结构生成
│   │   └── DAO/                             #   PyTorch 基础算子库
│   ├── AI4MD/                               # 机器学习分子动力学
│   │   ├── Lennard_Jones/                   #   LJ 力场算子 ✅
│   │   ├── GAFF2/                           #   GAFF2 力场算子 ✅
│   │   ├── PME/                             #   PME 静电算子 ✅
│   │   ├── SHAKE/                           #   SHAKE 约束算子 ✅
│   │   ├── velocity-verlet/                 #   Velocity Verlet 积分 ✅
│   │   └── Dissipative_particle_dynamics/   #   DPD 耗散粒子动力学 ✅
│   └── AI4PDE/                              # AI for PDE
│       ├── pinn/                            #   PINN 算子 ✅
│       ├── fno/                             #   FNO 算子 ✅
│       ├── deeponet/                        #   DeepONet 算子 ✅
│       └── pde_common/                      #   公共工具模块
├── prediction/                              # 预测优化方向
│   └── SmallData/                           # 小数据预测优化模型
│       ├── kernels.py                       #   GP 核函数参考实现 🔧
│       └── gpr.py                           #   GP 回归 + BO 参考实现 🔧
├── template/                                # 算子贡献模板
│   ├── algorithm.md                         #   算法说明模板
│   ├── references.md                        #   参考文献模板
│   ├── operator_example.md                  #   算子代码框架模板
│   └── test_architecture.md                 #   测试架构模板
├── roadmap.md                               # 开发路线图
├── FAQ.md                                   # 常见问题
└── AGENTS.md                                # 开发辅助规则
```

## 重点关注领域

### 科学计算

| 子方向 | 状态 | 目标 | 关键算子/模型类型 |
|--------|------|------|-------------------|
| **材料性质预测与结构生成** | 🔧 PyTorch 参考 | 基于第一性原理数据构建材料性质快速预测与逆向生成能力 | 晶体图神经网络、材料描述符计算、晶体结构生成模型、相图预测等 |
| **机器学习分子动力学** | ✅ Ascend C 就绪 | 以 ML 势替代经典力场，实现 DFT 精度下的大体系长时间 MD 模拟 | DeepMD/SchNet/MACE 推理算子、描述符在线计算、D3 色散校正等 |
| **AI for PDE** | ✅ Ascend C 就绪 | 用 AI 方法加速或替代传统 PDE 求解器，覆盖流体/传热/结构力学 | PINN/FNO/DeepONet 推理算子等 |

### 预测优化

| 子方向 | 状态 | 目标 | 关键算子/模型类型 |
|--------|------|------|-------------------|
| **小数据预测优化模型** | 🔧 PyTorch 参考 | 解决标记数据稀少的化工场景，聚焦小样本/主动学习/贝叶斯优化 | 高斯过程回归、贝叶斯神经网络、贝叶斯优化、度量学习等 |

> ✅ Ascend C 就绪 = 已完成 Ascend C 算子开发，含完整测试 | 🔧 PyTorch 参考 = 提供 PyTorch 参考实现，可作为 Ascend C 迁移基础

> 详细算子规划请参见 [roadmap.md](roadmap.md)。

## 贡献模板

为降低新算子贡献门槛，仓库提供了标准模板，位于 [template/](template/) 目录：

| 模板 | 说明 |
|------|------|
| [algorithm.md](template/algorithm.md) | 算法说明文档模板 —— 涵盖算法原理、NPU 实现、精度与性能分析 |
| [references.md](template/references.md) | 参考文献模板 —— 分类整理基础理论、硬件实现、领域应用和第三方实现 |
| [operator_example.md](template/operator_example.md) | 算子示例模板 —— 目录结构、Host/Kernel 代码框架、构建配置 |
| [test_architecture.md](template/test_architecture.md) | 测试架构模板 —— C++ UT、Python 集成测试、性能基准测试规范 |

## 维护团队

### Maintainer列表
- 黄剑兴  huangjianxing4@huawei.com
- 张玉橙 zhangyucheng23@huawei.com
- 周吉彬 zhoujibin@dicp.ac.cn
- 高菲 gaofei06@petrochina.com.cn

### Committer列表
- 刘非 liuf23357@gmail.com
- 刘海东 aliutec@163.com
- 高梓博 gaozibo@petrochina.com.cn
- 马博文  iambowen.m@qq.com
- 刘达林 liudalin@huawei.com
- 赵俊 roomdream@qq.com
- 郑柳琪 2557692481@qq.com
- 张强豪 1964035193@qq.com
- 李姝漫 1404537011@qq.com

## 参考资源

为帮助开发者学习 Ascend C 算子开发，建议参考以下 CANN 官方仓库（可 clone 到本地 `Reference/` 目录，该目录不参与版本管理）：

### 基础数学库

| 仓库 | 说明 |
|------|------|
| [opbase](https://atomgit.com/cann/opbase) | Ascend C 算子开发基础库，提供通用宏、接口与工具 |
| [ops-math](https://atomgit.com/cann/ops-math) | 数学运算算子库（Element-wise、Reduction 等） |
| [ops-blas](https://atomgit.com/cann/ops-blas) | BLAS 线性代数算子库（MatMul、Gemm 等） |
| [ops-fft](https://atomgit.com/cann/ops-fft) | 傅里叶变换算子库 |
| [ops-sparse](https://atomgit.com/cann/ops-sparse) | 稀疏计算算子库 |
| [ops-tensor](https://atomgit.com/cann/ops-tensor) | 张量操作算子库（Reshape、Slice、Concat 等） |
| [ops-nn](https://atomgit.com/cann/ops-nn) | 神经网络算子库（Conv、Pool、Activation 等） |
| [ops-rand](https://atomgit.com/cann/ops-rand) | 随机数生成算子库 |

### 开发标准与调试工具

| 仓库 | 说明 |
|------|------|
| [cann-learning-hub](https://atomgit.com/cann/cann-learning-hub) | CANN 学习中心，含教程、示例与最佳实践 |
| [oam-tools](https://atomgit.com/cann/oam-tools) | 算子开发调试与性能分析工具集 |
| [ops-test-kit](https://atomgit.com/cann/ops-test-kit) | 算子测试套件，用例生成与结果分析 |
| [pypto](https://atomgit.com/cann/pypto) | PyTorch 算子自动迁移工具（PyTorch → Ascend C） |

### AI Kernel 生成参考

| 仓库 | 说明 |
|------|------|
| [cann-bench](https://atomgit.com/cann/cann-bench) | CANN 性能基准测试与 Benchmark |
| [cannbot-skills](https://atomgit.com/cann/cannbot-skills) | CANN 开发辅助 Skills 与自动化工具 |
| [cann-samples](https://atomgit.com/cann/cann-samples) | Ascend C 算子开发完整示例集 |

### 相关参考库

| 仓库 | 说明 |
|------|------|
| [cann-ops-competitions](https://atomgit.com/cann/cann-ops-competitions) | CANN 算子竞赛参赛作品集 |
| [asnumpy](https://atomgit.com/cann/asnumpy) | NumPy 兼容 API，方便从 NumPy 迁移到 Ascend |
| [ascend-boost-comm](https://atomgit.com/cann/ascend-boost-comm) | Ascend Boost 通信库 |

> 可通过 `git clone` 将上述仓库下载至本地 `Reference/` 目录作为学习参考。该目录已加入 `.gitignore`，不会纳入本仓库版本管理。

## 快速上手

环境要求
- CANN ≥ 7.0
- Atlas A2/A3 训练/推理卡
- CMake ≥ 3.16

## 许可证

本仓库以生态繁荣为目标，选取 [Apache 2.0](https://www.apache.org/licenses/) 许可证。

## 算子贡献指南

详情参考公告 [算子贡献指南 1.0](https://gitcode.com/cann/mat-chem-sim-pred/discussions/1)

### 提交流程：

Fork 仓库 → 开发测试 → 提交 PR → SIG Review → 合入