# 常见问题（FAQ）

> 本 FAQ 侧重具体的技术问题和实操疑点，仓库定位、样例列表等基础信息请参见 [README.md](README.md)。

---

## 算子开发

### 我开发了一个新算子，目录结构和代码规范上需要注意什么？

参考 [template/](template/) 下的四个模板文件，分别对应算法说明、参考文献、算子代码框架、测试架构。特别注意：

- 算子目录名统一使用小写+下划线，如 `my_operator/`
- Host 侧 API 命名遵循 `aclnn{算子名}` 约定
- Kernel 入口函数使用 `__aicore__` 修饰符
- Tiling 参数需显式定义结构体，不能硬编码
- 所有数学函数（`sin`、`cos`、`sqrt` 等）需使用位运算 + Newton-Raphson 迭代替代，不能依赖 `<math.h>`

### Ascend C 算子中能否调用标准数学库（如 `<math.h>`）？

不能。Ascend C Kernel 侧不直接支持 `<math.h>` 和 `__builtin_*` 函数。所有数学函数需自行实现，推荐使用位操作 + Newton-Raphson 迭代（2 次迭代即可达到约 1×10⁻⁷ 精度）。已有 GAFF2、SHAKE 等算子的实现可作为参考。

### PyTorch 算子与 Ascend C 算子在此仓库中如何共存？

两种形态均可提交。当前仓库以 Ascend C 自定义算子为主（位于 `op_host/`、`op_kernel/`），但每个算子配套的 Python 测试（`tests/test_*.py`）均基于 PyTorch 实现了参考计算逻辑，可直接作为 PyTorch → CANN 迁移的参考。如果你希望贡献纯 PyTorch 的模型样例（不涉及 Ascend C Kernel），放在对应子目录下并附上 README 即可。

### 我需要为算子提供哪些测试？

至少提供两类测试：

1. **C++ 单元测试**（`tests/ut/op_kernel/test_*.cpp`）：与 CPU 参考实现逐结果对比
2. **Python 集成测试**（`tests/test_*.py`）：端到端调用，对比 NumPy 和 PyTorch 结果

性能基准测试（`tests/benchmark_*.py`）虽然不是必须，但非常推荐。

---

## 计算精度

### NPU 上 float32 精度足够做分子动力学模拟吗？

已有算子的实测数据显示，在 Ascend C 上采用 float32 配合位运算数学函数实现，相对误差约 1×10⁻⁷，对 MD 模拟可忽略。SHAKE 约束在 float32 下经过 10 步 VV+SHK 循环后，坐标最大偏差约 9.54×10⁻⁶ nm（相对误差 < 0.01%）。

但对于需要长期能量守恒的模拟（如 NVE 系综），建议对照 CPU double 结果做更长时间的稳定性验证。

### 力场算子的力约定是什么？

所有力的定义为 $F = -\nabla E$（指向能量降低方向），与 GROMACS 约定一致。返回力单位为 $\text{kJ}/(\text{mol} \cdot \text{nm})$。

---

## 部署与集成

### 这些算子能否直接替换 GROMACS/LAMMPS 中的对应计算模块？

当前算子以 Ascend C 自定义算子的形式提供，提供了 aclnn 风格的 C++ API，可以通过 ACL（Ascend Computing Language）集成到自定义 MD 框架中。但尚未提供直接的 GROMACS/LAMMPS patch，如需在标准 MD 包中使用，需自行编写中间适配层。

### 算子如何在多卡或多节点上使用？

当前算子的实现是单设备（单 NPU）粒度的。如果需要多卡或多节点并行，需要在外部框架层（如自定义的 MPI + Python 调度）实现数据拆分和通信。CANN 提供了 HCCL（集合通信库）可用于多设备场景，但本仓库暂未提供多卡版的算子样例。

### 算子能否与 PyTorch 的 autograd 联动？

可以。你可以在 PyTorch 中通过 `torch.ops` 或自定义 `torch.autograd.Function` 封装 Ascend C 算子的 ACL 调用，从而在前向推理中利用 NPU 加速，同时保留 PyTorch 的自动微分能力。这对于需要可微分 MD、PINN 等场景非常有用。

---

## 性能分析

### 在 Atlas A2/A3 上运行这些算子，性能能达到什么水平？

以 Lennard-Jones 力场算子为例（Atlas A3 环境）：

| 原子数 | CPU (PyTorch) | NPU 融合算子 | 加速比 |
|--------|---------------|-------------|--------|
| 64 | 0.54 ms | 0.57 ms | 0.96× |
| 128 | 25.57 ms | 0.75 ms | 34.21× |
| 256 | 174.96 ms | 0.85 ms | 206.23× |
| 512 | 183.00 ms | 1.45 ms | 126.36× |

在工业规模的体系（数千至数万原子）上，融合算子将古典力场计算从 CPU 上的"瓶颈步骤"变为 NPU 上的"几乎免费步骤"。但注意，跨场景的性能数据差异较大，建议针对自己的体系做基准测试。

### 我应该如何对我的算子做性能分析？

每个算子的 `tests/` 目录下提供了 `benchmark_*.py` 脚本，作为性能基准测试的起点。建议至少测试 3-5 个不同规模（如小/中/大/超大），并记录：

- NPU Kernel 执行时间（通过 ACL 的 Event 计时）
- Host-Device 数据拷贝时间
- 与 CPU 参考实现的加速比
- 与 PyTorch GPU（如有）的加速比

---

## 兼容性与限制

### 算子是否支持动态 shape？

当前已发布算子（LJ、GAFF2、PME、SHAKE、VV）均支持动态原子数输入（通过参数传递 `numAtoms`），但不支持动态的 batch 维度。如果需要批次处理，可在 Host 侧循环调度多个独立的 Kernel 任务。

### 是否支持混合精度计算？

当前算子全部采用 float32 精度计算。CANN 工具链支持 FP16/INT8 量化，但科学计算场景对精度要求较高，暂未启用混合精度。未来规划中会评估在特定环节（如非键力计算中的截断判断、PME 倒易空间 FFT）引入半精度加速的可能性。

### 算子是否支持 Windows 开发环境？

算子本身在昇腾 NPU 上运行，开发环境需要在 Linux 服务器上配置 CANN 工具链。Windows 可用于代码编辑和文档编写，但编译和运行测试需要在 Linux 环境下完成。建议通过远程开发（VS Code Remote / SSH）方式连接到 Atlas 服务器。

---

## 贡献与交流

### 我发现的 bug 或想提出的功能建议应该发在哪里？

- **Bug 报告** → [Issues](https://gitcode.com/cann/mat-chem-sim-pred/issues)
- **功能建议/新算子需求** → [讨论区](https://gitcode.com/cann/mat-chem-sim-pred/discussions) 的 "创意想法" 分类
- **使用问题/求助** → [讨论区](https://gitcode.com/cann/mat-chem-sim-pred/discussions) 的 "问答求助" 分类

### 我想贡献的算子不在 roadmap 上，可以提交吗？

可以。roadmap 列出的是团队优先关注的方向，但社区贡献不受 roadmap 限制。只要算子符合"材料/化工/钢铁/油气等工业场景中的模拟仿真或预测优化"这一主题，且提供了完整文档和测试，即可提交 PR。建议先在讨论区发起话题，获取反馈后再开始编码。

### 我如何在自己的文章中引用这个仓库或其中的算子？

建议引用格式：

```
mat-chem-sim-pred: CANN-based operator library for material and chemical simulation and prediction.
https://gitcode.com/cann/mat-chem-sim-pred
```

引用特定算子时，附上该算子的算法说明文档链接（`docs/algorithm.md`）和对应的 DOI（如有）。

### 维护团队如何分工？

- **Maintainer**：负责仓库整体方向、SIG Review 和合入决策
- **Committer**：负责日常代码审查、Issue 回复和 CI 维护

如有紧急问题，可直接 @Maintainer 列表中的成员。