# mat-chem-sim-pred

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/mat-chem-sim-pred)

本项目针对材料、化工、钢铁、油气等工业研发、生产业务中的模拟仿真类和预测优化类典型模型、加速算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地基于CANN平台使用对应模型。

---

## 目录

- [背景](#背景)
- [仓库结构](#仓库结构)
- [样例列表](#样例列表)
- [快速上手](#快速上手)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 背景

工业研发与生产场景中广泛涉及**模拟仿真**与**预测优化**两类计算任务：

- **模拟仿真类**：包括分子动力学模拟、材料性质计算、流体力学仿真、反应-扩散过程建模等，计算量大、并行化需求高，是CANN异构计算加速的重点方向。
- **预测优化类**：包括表格数据预测、时序数据建模、小样本优化、贝叶斯优化等，面向工业研发中的配方设计、工艺参数优化、质量预测等实际问题。

本仓库将上述场景中的典型模型与算法封装为基于CANN（Ascend C）的高性能算子样例，覆盖从科学计算到数据驱动的全链路，帮助开发者快速在昇腾NPU上部署和使用。

## 仓库结构

```
mat-chem-sim-pred/
├── simulation/                              # 模拟仿真类样例
│   ├── MaterialPropertyPrediction/          # 材料性质预测与材料结构生成
│   ├── AI4MD/                               # 机器学习分子动力学（含经典力场算子）
│   └── AI4PDE/                              # AI for PDE
├── prediction/                              # 预测优化类样例
│   ├── TabularData/                         # 表格类数据预测预训练模型
│   ├── TimeSeries/                          # 时序类数据预测预训练模型
│   └── SmallData/                           # 小数据预测优化模型
├── template/                                # 算子贡献模板
└── roadmap.md                               # 算子开发路线图
```

## 样例列表

### 模拟仿真类

| 子方向 | 样例 | 状态 | 说明 |
|--------|------|------|------|
| **分子动力学** | Lennard-Jones 力场融合计算 | ✅ 已发布 | LJ 12-6 势能与力计算 |
| **分子动力学** | GAFF2 力场 | ✅ 已发布 | 键/角/二面角/LJ/库仑全项 |
| **分子动力学** | PME 长程静电 | ✅ 已发布 | Ewald 求和 + PME 加速 |
| **分子动力学** | SHAKE 约束 | ✅ 已发布 | 迭代键长约束 |
| **分子动力学** | Velocity Verlet 积分器 | ✅ 已发布 | 3 步时间积分 + NPT 恒温恒压 |
| **分子动力学** | 耗散粒子动力学（DPD） | ✅ 已发布 | 介观尺度流体模拟 |
| **机器学习分子动力学** | ML 势推理算子（DeepMD/SchNet/MACE 等） | 📋 规划中 | 见 [roadmap.md](roadmap.md) |
| **AI for PDE** | PINN / FNO / DeepONet 推理算子 | 📋 规划中 | 见 [roadmap.md](roadmap.md) |
| **材料性质预测** | 晶体图神经网络 / 描述符计算 / 结构生成 | 📋 规划中 | 见 [roadmap.md](roadmap.md) |

### 预测优化类

| 子方向 | 样例 | 状态 | 说明 |
|--------|------|------|------|
| **表格数据** | TabTransformer / FT-Transformer / TabNet 推理算子 | 📋 规划中 | 见 [roadmap.md](roadmap.md) |
| **时序数据** | TimesNet / PatchTST / TimesFM / Mamba 推理算子 | 📋 规划中 | 见 [roadmap.md](roadmap.md) |
| **小数据优化** | 高斯过程回归 / 贝叶斯优化 / 主动学习 / 小样本学习 | 📋 规划中 | 见 [roadmap.md](roadmap.md) |

> 详细算子规划与算法说明请参见 [roadmap.md](roadmap.md)。

## 快速上手

### 环境要求

- CANN ≥ 7.0
- Atlas A2/A3 训练/推理卡
- CMake ≥ 3.16

### 构建与运行

以 Lennard-Jones 力场算子为例：

```bash
cd simulation/AI4MD/Lennard_Jones
mkdir build && cd build
cmake ..
make
./test_lj_force
```

各算子目录下均提供独立的构建说明，详见对应 `README.md`。

## 贡献指南

### 常见问题

首次贡献前建议阅读 [FAQ.md](FAQ.md)，了解仓库定位、算子形态要求（Ascend C 与 PyTorch 算子均欢迎）和常见技术问题。

### 贡献模板

为降低新算子贡献门槛，仓库提供了标准模板，位于 [template/](template/) 目录：

| 模板 | 说明 |
|------|------|
| [algorithm.md](template/algorithm.md) | 算法说明文档模板 —— 涵盖算法原理、NPU 实现、精度与性能分析 |
| [references.md](template/references.md) | 参考文献模板 —— 分类整理基础理论、硬件实现、领域应用 |
| [operator_example.md](template/operator_example.md) | 算子示例模板 —— 目录结构、Host/Kernel 代码框架、构建配置 |
| [test_architecture.md](template/test_architecture.md) | 测试架构模板 —— C++ UT、Python 集成测试、性能基准测试规范 |

### 提交流程

Fork 仓库 → 开发测试 → 提交 PR → SIG Review → 合入

详情参考公告 [算子贡献指南 1.0](https://gitcode.com/cann/mat-chem-sim-pred/discussions/1)

### 维护团队

**Maintainer**
- 黄剑兴 huangjianxing4@huawei.com
- 张玉橙 zhangyucheng23@huawei.com
- 周吉彬 zhoujibin@dicp.ac.cn
- 高菲 gaofei06@petrochina.com.cn

**Committer**
- 刘非 liuf23357@gmail.com
- 刘海东 aliutec@163.com
- 高梓博 gaozibo@petrochina.com.cn
- 马博文 iambowen.m@qq.com
- 刘达林 liudalin@huawei.com
- 赵俊 roomdream@qq.com
- 郑柳琪 2557692481@qq.com
- 张强豪 1964035193@qq.com
- 李姝漫 1404537011@qq.com

## 许可证

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)