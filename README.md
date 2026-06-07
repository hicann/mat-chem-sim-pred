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
│   ├── AI4MD/                               # 机器学习分子动力学
│   └── AI4PDE/                              # AI for PDE
├── prediction/                              # 预测优化方向
│   ├── TabularData/                         # 表格类数据预测预训练模型
│   ├── TimeSeries/                          # 时序类数据预测预训练模型
│   └── SmallData/                           # 小数据预测优化模型
├── template/                                # 算子贡献模板
└── roadmap.md                               # 算子开发路线图
```

## 重点关注领域

### 科学计算

| 子方向 | 目标 | 关键算子/模型类型 |
|--------|------|-------------------|
| **材料性质预测与结构生成** | 基于第一性原理数据构建材料性质快速预测与逆向生成能力 | 晶体图神经网络、材料描述符计算、晶体结构生成模型、相图预测等 |
| **机器学习分子动力学** | 以 ML 势替代经典力场，实现 DFT 精度下的大体系长时间 MD 模拟 | DeepMD/SchNet/MACE 推理算子、描述符在线计算、D3 色散校正等 |
| **AI for PDE** | 用 AI 方法加速或替代传统 PDE 求解器，覆盖流体/传热/结构力学 | PINN/FNO/DeepONet/ MeshGraphNet 推理算子等 |

### 预测优化

| 子方向 | 目标 | 关键算子/模型类型 |
|--------|------|-------------------|
| **表格类数据预训练模型** | 针对化工配方/物性/操作条件表，构建通用预训练 + 微调框架 | TabTransformer、FT-Transformer、TabNet、Gradient Boosting 替代等 |
| **时序类数据预训练模型** | 面向 DCS 历史数据/传感器流/批次轨迹的时序建模 | TimesNet、PatchTST、TimesFM、Mamba/SSM 等 |
| **小数据预测优化模型** | 解决标记数据稀少的化工场景，聚焦小样本/主动学习/贝叶斯优化 | 高斯过程回归、贝叶斯神经网络、贝叶斯优化、度量学习等 |

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