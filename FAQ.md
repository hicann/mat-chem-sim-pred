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

## 架构差异

### Ascend NPU 与 NVIDIA GPU 的计算架构有何本质区别？

NVIDIA GPU 采用 **SIMT**（Single Instruction, Multiple Thread）架构，而 Ascend NPU 采用 **SIMD**（Single Instruction, Multiple Data）为主的架构。这是两种截然不同的并行计算模型：

| 维度 | NVIDIA GPU (SIMT) | Ascend NPU (SIMD) |
|------|-------------------|-------------------|
| **执行模型** | 数千个轻量线程并行，每个线程有独立 PC | 向量化指令驱动，数据级并行（Vector/Cube Unit） |
| **线程分歧** | 允许（warp 内 divergence 会导致序列化） | 不存在线程概念，天然无分歧 |
| **编程思维** | 思考 "N 个线程各处理 1 个元素" | 思考 "一条指令处理 N 个元素" |
| **数据搬运** | 隐含在 Load/Store 指令中 | 显式搬运（GM → Local Memory → 计算 → 写回），通过 Pipe 机制流水 |
| **适合场景** | 不规则并行（稀疏、图、动态控制流） | 规则并行（矩阵乘法、卷积、向量化计算） |
| **开发语言** | CUDA | Ascend C |

### 这对算子开发有什么实际影响？

**对于从 CUDA 迁移到 Ascend C 的开发者**，需要转变几个核心思维：

1. **不要逐元素编程**：CUDA 中你写 `kernel<<<grid, block>>>(data)` 启动 N 个线程各算一个元素；Ascend C 中你操作的是向量寄存器（如 `LocalTensor<float>`），一条指令处理 64 或 128 个元素。

2. **数据搬运需显式管理**：Ascend C 的 `DataCopy` / `Pipe` 机制要求开发者手动控制 GM（全局内存）→ Local Memory（片上缓存）的数据流动，这与 CUDA 中相对透明的 cache 层级不同。

3. **Tiling 是关键**：由于 Local Memory（约 2MB）远小于显存，大数组需要切分成 tile 逐块处理。本仓库的 LJ、GAFF2、DPD 等算子的 `tiling` 文件是很好的参考。

4. **向量化为王**：尽量将标量运算改写为向量运算。例如计算 `y = 1.0 / sqrt(x)`，不要写循环逐元素处理，而应调用 `Reciprocal` + `Sqrt` 向量指令。

### Ascend C 的 Pipe 机制是什么？

Pipe 是 Ascend C 提供的数据搬运-计算流水线机制，允许数据加载、计算、存储三个阶段重叠执行：

```
  ┌─────────┐      ┌─────────┐      ┌─────────┐
  │ 数据加载 │ ──→  │   计算   │ ──→  │ 结果回写 │
  │ (GM→LM) │      │ (Vector │      │ (LM→GM) │
  │         │      │  /Cube) │      │         │
  └─────────┘      └─────────┘      └─────────┘
       ↑                 ↑                ↑
       └───────── 流水线并行 ──────────────┘
```

通过 `TPipe` 和 `TQue`（队列）管理不同阶段的 buffer，可实现接近理论峰值的计算吞吐。当前 AI4MD 和 AI4PDE 的多个算子均利用了 Pipe 机制进行性能优化。

---

## 计算精度

### NPU 上 float32 精度足够做分子动力学模拟吗？

已有算子的实测数据显示，在 Ascend C 上采用 float32 配合位运算数学函数实现，相对误差约 1×10⁻⁷，对 MD 模拟可忽略。SHAKE 约束在 float32 下经过 10 步 VV+SHK 循环后，坐标最大偏差约 9.54×10⁻⁶ nm（相对误差 < 0.01%）。

但对于需要长期能量守恒的模拟（如 NVE 系综），建议对照 CPU double 结果做更长时间的稳定性验证。

### 部分科学计算需要 FP64 双精度，但 Ascend 910B/910C 只支持到 FP32，怎么办？

这是一个**关键限制**，需要开发者充分了解：

**硬件现状**：
- Ascend 910B / 910C 的 AI Core（Cube Unit + Vector Unit）**原生仅支持 FP32**，没有像 NVIDIA GPU（V100 起支持 FP64 Tensor Core，A100 双精度达到 9.7 TFLOPS）那样的硬件 FP64 算力
- Ascend 950 系列在架构设计上是否有 FP64 支持，需以华为官方发布的规格为准
- 部分科学计算场景（如 DFT 中的对角化、MD 中的长时间积分、PDE 中的刚性方程求解）在 FP32 下可能出现精度不足

**影响评估**：

| 场景 | FP32 风险 | 建议 |
|------|-----------|------|
| 短时 MD 能量/力预测（< 1 ns） | 低（实测误差 < 1×10⁻⁶） | 可直接使用 |
| 长期 NVE 系综（> 10 ns） | 中（能量漂移可能累积） | 需与 FP64 基准对比验证 |
| 结构优化 / 过渡态搜索 | 中高（梯度精度影响收敛） | 建议用 CPU FP64 做最终验证 |
| 高精度电子结构计算 | 高（本征值精度不够） | 建议使用 CPU/GPU FP64 方案 |
| AI for PDE 逆问题 | 低（网络本身精度受限） | 可直接使用 |

**实用建议**：
1. **场景决定精度需求**：如果你的场景只需要"相对正确"的趋势（如 AI 势函数训练、材料筛选排序、工艺参数优化），FP32 完全够用
2. **必须做精度验证**：在切换到 NPU 之前，用 CPU double 结果作为基准，在你关注的全部物理量上做逐点对比
3. **混合精度策略**：可在 Host 侧用 CPU double 做累积误差敏感的部分（如长时间积分），NPU 上做大规模并行计算部分
4. **社区交流**：如果你在 FP32 下遇到了精度问题，欢迎在讨论区分享你的场景，也欢迎有 FP64 需求的用户一起推动 CANN 工具链的改进

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

### Ascend 910B、910C 和 950 系列对本仓库算子的支持情况如何？

| 硬件系列 | 产品型号 | 与本仓库算子的兼容性 | 说明 |
|----------|----------|---------------------|------|
| **Atlas A2** | Ascend 910B | ✅ 完全支持 | 当前开发和测试的主力平台，`SOC_VERSION` 设为 `Ascend910B1` 即可 |
| **Atlas A3** | Ascend 910C | ✅ 完全支持（已验证） | LJ 力场算子的性能数据即在 Atlas A3 上测得，CMake 中指定对应 `SOC_VERSION` |
| **Ascend 950 系列** | 待发布 | ❓ 待验证 | 作为昇腾下一代架构，预计提供更高算力和新指令集。算子级兼容性需等待硬件就绪后测试 |
| **Atlas 200/500 A2** | Ascend 310B | ⚠️ 部分支持 | 推理卡，算力受限。小规模体系可运行，大规模 MD 模拟建议使用 A2/A3 |
| **Atlas 800T A2** | Ascend 910B (训练) | ✅ 完全支持 | 训练服务器场景，与 A2 训练卡算子兼容 |

**关键注意事项**：

1. **SOC_VERSION 必须匹配**：编译时需通过 `-DSOC_VERSION=` 指定正确的芯片型号，不同系列使用的指令集和编译器优化策略不同
2. **性能预期差异**：同一算子在 910B 和 910C 上的性能可能不同（缓存大小、频率差异），建议在目标硬件上重新做基准测试
3. **950 系列前瞻**：虽然当前算子尚未在 950 上验证，但 Ascend C 作为昇腾统一的算子开发语言，具备良好的前向兼容性。后续硬件就绪后，本仓库会及时跟进适配
4. **驱动和 CANN 版本要求**：
   - 910B：CANN ≥ 7.0，驱动 ≥ 23.0
   - 910C：CANN ≥ 7.0.RC1，驱动 ≥ 24.0
   - 具体版本兼容性矩阵请查阅华为官方文档

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