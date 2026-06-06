# mat-chem-sim-pred

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/mat-chem-sim-pred)

## 项目简介
本仓库为基于华为CANN计算框架开发的化工行业专用算子库，聚焦科学计算与预测优化场景。通过高性能算子加速分子模拟、反应路径预测、物质特性分析等核心任务，为化工研发、工艺优化及智能制造提供高效计算能力支撑。

## 已发布算子
| 算子               | 功能                  | 适用场景                  | 状态  |
| ---------------- | ------------------- | --------------------- | --- |
| **LJForceFused** | Lennard-Jones力场融合计算 | 分子动力学模拟（催化剂设计、材料研发等）  | ✅已发布 |
| **DPD** | 耗散粒子动力学  | 介观尺度流体模拟（聚合物加工、油田化学等） | ✅已发布 |
| **GAFF2** | GAFF2 力场势能+力计算 | 键伸缩、键角弯曲、二面角扭转、Lennard-Jones 12-6、库仑静电，含 PBC 最小镜像和 1-4 对缩放 |✅ 已发布 |
| **PME** | Ewald求和 + PME加速的长程静电计算 | 实空间 Ewald（erfc 衰减）+ 倒易空间 PME（B-spline 插值 + 3D FFT）+ 自能修正 + 分子内排除 |✅已发布 |
| **SHAKE** | 迭代键长约束算法 | 固定含氢键长，与 Velocity Verlet 配合使用，收敛容差 1×10⁻⁴ nm | ✅已发布 |
| **Velocity Verlet** | 标准 3 步时间积分器 | vv_integrate（半步速度 + 全步位置）、vv_finish（速度完成）、thermo_scale（NPT 恒温/恒压缩放）| ✅已发布 |

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

## 许可证选择
本仓库以生态繁荣为目标，选取[Apache 2.0](https://www.apache.org/licenses/)许可证。

## 算子贡献指南
详情参考公告 [算子贡献指南1.0](https://gitcode.com/cann/mat-chem-sim-pred/discussions/1)

## 提交流程：
Fork仓库 → 开发测试 → 提交PR → SIG Review → 合入





