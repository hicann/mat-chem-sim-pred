# 第一批 10 个时序算子 TorchAir 与自定义算子验证

## 1. 验证目的

本报告验证第一批 10 个 Ascend C 算子的原始 `torch_npu` 框架路径能否被 TorchAir 自动优化，并比较完整图与自定义算子的稳定运行性能。

必须区分两种图模式结果：

1. **原模型直接成图**：不修改模型数学表达，只使用 `torch.compile(backend="npu", fullgraph=True, dynamic=False)`。
2. **人工等价改写后成图**：先把当前 TorchAir 不支持的 ATen 表达改写成数学等价的基础张量操作，再执行完整图编译。这代表经过人工优化的强框架基线，不代表 TorchAir 能自动优化原模型。

主要性能对照均在同一 Ascend 910B3 环境完成。`custom` 时间包含 ACLNN 描述符、执行器、kernel 和同步开销。

`BatchSpdInvFp32` 是唯一例外：Koopa 原生 `torch.linalg.lstsq` 不支持 NPU，并回退 CPU，TorchAir 也无法捕获，因此该项按 CPU fallback 与自定义全设备路径对比。

## 2. 环境与判定标准

| 项目 | 配置 |
|---|---|
| NPU | Ascend 910B3 |
| CANN | 8.1.RC1 |
| PyTorch | 2.5.0 |
| torch_npu | 2.5.1 |
| 图后端 | `npu`，即当前软件栈的 TorchAir 后端 |
| shape | 固定 shape，`dynamic=False` |
| 成图要求 | `fullgraph=True`；graph break 或 converter 缺失直接判定失败 |

本报告只代表上述软件栈，不能直接外推到后续版本。

## 3. 主测试 shape

| 算子组 | 主测试 shape |
|---|---|
| UnICORNN、SRU、S6、CfC、coRNN、LTC | `B=32,L=336,IN=11,H=64`；LTC 另有 `K=6` |
| SelectiveScan1D | `B=1,L=1024,D=1536,N=16` |
| TiRexSlstmCell | `B=64,S=64,H=512,heads=4` |
| AutoCorrFusedAggregate | `B=1,heads=4,embed=16,L=192,top_k=3` |
| BatchSpdInvFp32 | `B=32,m=7,E=128`；另有 Koopa `seq336,pred96,B32` E2E |

`B` 是批量，`L/S` 是时间步数，`IN` 是输入特征数，`H` 是隐藏状态宽度，`D/N` 是 SelectiveScan 的通道数和状态维，`K` 是 LTC 每步内部 ODE 更新次数。

## 4. 原模型直接成图能力

| 模型 | 算子 | 原模型 fullgraph | 证据或失败原因 |
|---|---|---|---|
| Mamba | SelectiveScan1D | 支持 | 主 shape 18,441 个 FX 节点 |
| Mamba/S6 | S6ScanFused | 不支持 | 缺 `aten.softplus.default` converter |
| UnICORNN | UnicornnScanFused | 支持 | 主 shape 4,715 个 FX 节点 |
| SRU | SruScanFused | 支持 | 主 shape 7,069 个 FX 节点 |
| CfC | CfcScanFused | 支持 | 主 shape 7,062 个 FX 节点 |
| coRNN | CornnScanFused | 支持 | 主 shape 3,703 个 FX 节点 |
| LTC | LtcScanFused | 支持 | 主 shape 24,878 个 FX 节点 |
| TiRex/xLSTM | TirexSlstmCell | 不支持 | 缺 `aten.log_sigmoid_forward.default` converter |
| Autoformer | AutoCorrFusedAggregate | 不支持 | 缺 `aten.roll.default` converter |
| Koopa | BatchSpdInvFp32 | 不支持 | `linalg_lstsq` 不支持 NPU，并回退 CPU；Dynamo 也无法捕获 |

原模型直接成图汇总：`6/10` 支持，`4/10` 不支持。

## 5. 原模型可直接成图的 6 个算子

| 算子 | 主测试 shape | FX 节点 | 首次编译+运行 | eager | TorchAir | custom | 图/eager | custom/图 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| UnicornnScanFused | B32 L336 IN11 H64 | 4,715 | 250.93 s | 91.229 ms | 3.544 ms | 0.521 ms | 25.74x | 6.80x |
| SruScanFused | B32 L336 IN11 H64 | 7,069 | 165.43 s | 118.829 ms | 5.272 ms | 0.884 ms | 22.54x | 5.97x |
| CfcScanFused | B32 L336 IN11 H64 | 7,062 | 154.54 s | 118.244 ms | 8.090 ms | 1.265 ms | 14.62x | 6.40x |
| CornnScanFused | B32 L336 IN11 H64 | 3,703 | 92.83 s | 73.529 ms | 3.579 ms | 1.718 ms | 20.54x | 2.08x |
| LtcScanFused | B32 L336 IN11 H64 K6 | 24,878 | 616.61 s | 386.003 ms | 10.227 ms | 4.909 ms | 37.74x | 2.08x |
| SelectiveScan1D | B1 L1024 D1536 N16 | 18,441 | 883.51 s | 343.825 ms | 27.952 ms | 20.077 ms | 12.30x | 1.39x |

这 6 个算子的 custom 相对 TorchAir 仍快 `1.39x-6.80x`。TorchAir 将循环展开为 3,703 至 24,878 个 FX 节点，没有识别成固定大小的 scan kernel；首次编译和运行需要 `92.83-883.51 s`。

## 6. 人工等价改写后的 3 个强图基线

以下 3 个原模型表达均因当前 TorchAir 缺少对应 converter 而不能直接完整成图。为了给 custom 提供尽可能强的框架基线，测试人员先人工修改模型表达，再运行 TorchAir：

- AutoCorrelation：`roll` 改为静态 `slice + cat` 循环移位。
- S6：`softplus` 改为数值稳定的 `maximum + abs + exp + log1p`。
- TiRex：`logsigmoid(x)` 改为数值稳定的 `-softplus(-x)` 基础算子组合。

因此下表是“人工等价改写后的 TorchAir”对照，不是“原模型直接 TorchAir”对照。这也属于当前 TorchAir 软件栈的功能限制：使用框架图模式需要维护额外的模型改写代码。custom 算子自身直接实现目标子图语义，模型侧只需替换为 ACLNN 调用，不需要为 converter 缺失重写数学表达。

| 算子 | FX 节点 | 首次编译+运行 | 改写 eager | TorchAir | custom | 图/eager | custom/图 | custom max diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AutoCorrFusedAggregate | 1,546 | 69.70 s | 24.324 ms | 4.568 ms | 0.618 ms | 5.32x | 7.39x | `7.45e-09` |
| S6ScanFused | 7,069 | 141.83 s | 107.206 ms | 4.372 ms | 0.674 ms | 24.52x | 6.48x | `4.47e-08` |
| TirexSlstmCell | 2,888 | 158.61 s | 57.892 ms | 6.318 ms | 3.542 ms | 9.16x | 1.78x | `7.45e-08` |

AutoCorrelation 同时补充了宽动态范围输入验证，custom 对框架参考的最大误差为 `1.43e-05`。该结果使用改进后的 `ln2` 区间约简加六阶多项式指数近似。最终正负指数缩放分支在 910B3 重建后再次复验：模型量级 `0.610 ms / 7.45e-09`，宽动态范围 `0.633 ms / 1.43e-05`；表中 `0.618 ms` 保留同一轮 TorchAir/custom 配对计时，避免跨轮次混算倍率。

## 7. Koopa：NPU 能力补齐，不使用 TorchAir 性能口径

Koopa 原始 `torch.linalg.lstsq` 在当前软件栈中不支持 NPU，运行时明确回退 CPU。Dynamo 在 `aten.linalg_lstsq.default` 动态输出 shape 处失败，TorchAir 无法生成 NPU 完整图。

| 场景 | 原框架 CPU fallback | 自定义全设备路径 | speedup | 精度 |
|---|---:|---:|---:|---:|
| DMD, B32 m7 E128 | 4341.75 ms | 0.339 ms | 12823.56x | max diff `6.80e-08` |
| Koopa E2E, seq336 pred96 B32 | 5300.64 ms | 8.80 ms | 602.19x | max diff `2.38e-07` |

该算子的核心价值是补齐设备能力、消除 CPU fallback，使 Koopa 的 DMD 路径能够全设备运行。

## 8. 结论

1. TorchAir 能显著优化 6 个原模型循环，但没有自动生成固定大小的 scan kernel；custom 相对完整图仍快 `1.39x-6.80x`。
2. AutoCorrelation、S6 和 TiRex 在当前栈中因缺 converter，必须先人工改写模型表达才能使用 TorchAir。面对这一更强基线，直接实现目标语义的 custom 仍快 `1.78x-7.39x`，并避免维护额外改写代码和 `69.70-158.61 s` 的首次编译。
3. Koopa 原路径不支持 NPU，TorchAir 也无法捕获；BatchSpdInvFp32 的价值是提供全设备实现，Koopa E2E 相对 CPU fallback 提升 `602.19x`。
4. 性能数字必须注明对照对象：前 9 个算子对照 NPU eager/TorchAir，Koopa 单独对照 CPU fallback，不得混写。

结构化摘要见 [torchair_first_batch_results.json](torchair_first_batch_results.json)。各算子的 E2E、CPU oracle 和复现命令以其目录中的 `docs/benchmark.md` 和 `docs/test_report.md` 为准。
