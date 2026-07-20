# feature: add first-batch CANN operators for time-series forecasting models

## 类型

- [x] 新功能
- [x] 文档内容更新
- [x] 测试与 benchmark 交付物

## 提交身份

本次提交使用作者邮箱：

```text
aliutec@163.com
```

## 摘要

本 PR 新增第一批 10 个面向时序预测模型的 CANN/Ascend C 自定义算子，覆盖 Mamba/SSM、Autoformer、Koopa、TiRex/xLSTM、CfC、coRNN、SRU、UnICORNN 和 LTC 等模型或模型族。

这些算子解决两类问题：

1. 将时间维递归、状态扫描或 lag 聚合从框架侧的 Python 循环和多次小算子调用，改为单个 NPU kernel 内部推进，减少 kernel launch、框架调度和中间 Tensor 物化。
2. 补齐框架当前无法在 NPU 上执行的模型路径。Koopa 使用的 `torch.linalg.lstsq` 会回落到 CPU，`BatchSpdInvFp32` 将该关键计算替换为全程驻留 NPU 的实现。

除 Koopa 外，本 PR 的性能基线均为同一 Ascend NPU 上的 PyTorch/torch_npu 框架实现，而不是 CPU。进一步补充的 TorchAir `fullgraph=True` 测试表明：即使与图编译后的框架路径相比，自定义算子在正式 shape 下仍有 `1.39x-7.39x` 的稳态运行优势。

## 背景与价值

时序模型中的 scan、递归单元和 lag aggregation 具有明显的前后依赖：第 `t` 步需要读取第 `t-1` 步状态。框架通常会把每一步拆成若干独立 NPU 算子，因此一次序列推理可能产生数百到数万次小算子调度。单次计算量不大时，launch、同步、读写中间 Tensor 的成本会超过有效计算本身。

本 PR 将这些具有明确边界的热子图下沉为 Ascend C kernel，使中间状态保留在 kernel 内部或片上缓冲区中，并在一个 kernel 生命周期内完成时间维推进。该方式不改变模型公式和输出语义，模型接入层只需把目标子图替换为对应 ACLNN 调用。

## 变更范围

```text
prediction/ProcessControl/TimeSeriesForecast/
├── README.md
├── CANN_OPERATOR_LLM_PROMPT_TEMPLATE.md
├── PR_DESCRIPTION_first_batch_cann_ops.md
├── docs/
├── deliverables/
├── selective_scan_1d/
├── s6scan_fused/
├── autocorr_fused_aggregate/
├── batch_spd_inv_fp32/
├── tirex_slstm_cell/
├── cfc_scan_fused/
├── cornn_scan_fused/
├── sru_scan_fused/
├── unicornn_scan_fused/
└── ltc_scan_fused/
```

同时更新：

```text
prediction/ProcessControl/README.md
```

## 性能对比口径

性能数据按以下三种口径分别报告，不混合计算：

1. **历史 eager NPU 基线**：在 Ascend NPU 上使用 PyTorch/torch_npu 原始算子组合表达同一计算，包含时间循环和多次小算子 launch。该结果反映不开发自定义算子时的直接框架路径。
2. **TorchAir 强基线**：使用 `fullgraph=True`、`dynamic=False` 对完整框架子图编译，在 warmup 后比较稳态运行时延。首次图编译时间不计入下表运行时延。
3. **Koopa CPU fallback 基线**：`torch.linalg.lstsq` 当前不支持 NPU，原路径会回落 CPU。该项衡量的是从 unsupported/fallback 到全程 NPU 可执行的收益，不能与普通 NPU framework baseline 混写。

测试环境：Ascend 910B3、CANN 8.1.RC1、PyTorch 2.5.0、torch_npu 2.5.1。

## 算子与性能总表

| 算子 | 服务模型/模型族 | 历史 eager torch_npu 对比 | TorchAir 或 fallback 强基线 | 精度与执行范围 |
|---|---|---|---|---|
| `SelectiveScan1D` | Mamba / SSM | Mamba block `9.02x`；模型 validation loop `10.22x` | 原表达直接 fullgraph：`27.952/20.077 ms`，custom 快 `1.39x` | prediction max diff `1.431e-06`，MSE/MAE diff `0` |
| `S6ScanFused` | S6 / Mamba-style SSM | 组件 `136.56x`；3 层 encoder `44.15x` | 原表达缺少 `aten.softplus.default` converter；等价基础算子改写后 `4.372/0.674 ms`，custom 快 `6.48x` | 3 层 encoder max diff `5.22e-08`；图基线 diff `4.47e-08` |
| `AutoCorrFusedAggregate` | Autoformer | validation loop `10.06x`；聚合子图 `28.06x` | 原表达缺少 `aten.roll.default` converter；等价 `slice+cat` 改写后 `4.568/0.618 ms`，custom 快 `7.39x` | prediction max diff `5.96e-08`；图基线 diff `7.45e-09`，宽动态范围 `1.43e-05` |
| `BatchSpdInvFp32` | Koopa / DMD | 不适用：框架路径无法保持在 NPU | DMD：CPU fallback `4341.75 ms`，custom `0.339 ms`，`12823.56x`；Koopa 完整推理：`5300.64/8.80 ms`，`602.19x` | DMD max diff `6.80e-08`；Koopa prediction max diff `2.38e-07` |
| `TirexSlstmCell` | TiRex / xLSTM | sLSTM cell `14.31x`；完整 sLSTM layer `10.46x` | 原表达缺少 `aten.log_sigmoid_forward.default` converter；等价稳定公式改写后 `6.318/3.542 ms`，custom 快 `1.78x` | cell max diff 约 `1.45e-06`；图基线 diff `7.45e-08` |
| `CfcScanFused` | CfC / continuous-time RNN | 组件 `84.83x`；3 层 encoder `70.26x` | 原表达直接 fullgraph：`8.090/1.265 ms`，custom 快 `6.40x` | encoder max diff `2.33e-07`；图基线 diff `2.38e-07` |
| `CornnScanFused` | coRNN | 组件 `56.73x`；3 层 encoder `54.15x` | 原表达直接 fullgraph：`3.579/1.718 ms`，custom 快 `2.08x` | encoder max diff `1.49e-07`；图基线 diff `1.04e-07` |
| `SruScanFused` | SRU | 组件 `163.48x`；3 层 encoder `71.46x` | 原表达直接 fullgraph：`5.272/0.884 ms`，custom 快 `5.97x` | encoder max diff `1.04e-07`；图基线 diff `1.79e-07` |
| `UnicornnScanFused` | UnICORNN | 组件 `160.68x`；3 层 encoder `100.43x` | 原表达直接 fullgraph：`3.544/0.521 ms`，custom 快 `6.80x` | encoder max diff `2.68e-07`；图基线 diff `1.90e-07` |
| `LtcScanFused` | LTC / continuous-time RNN | 组件 `82.44x`；3 层 encoder `76.90x` | 原表达直接 fullgraph：`10.227/4.909 ms`，custom 快 `2.08x` | encoder max diff `2.53e-07`；图基线 diff `1.34e-07` |

表中 `框架/custom ms` 均为同一正式 shape 下的稳态时延。详细 shape、warmup、重复次数和原始结果见 [`docs/torchair_first_batch_report.md`](docs/torchair_first_batch_report.md) 及各算子的 `docs/benchmark.md`、`docs/test_report.md`。

## TorchAir 结论说明

- `SelectiveScan1D`、`CfcScanFused`、`CornnScanFused`、`SruScanFused`、`UnicornnScanFused`、`LtcScanFused` 共 6 个原模型表达无需改写即可直接 `fullgraph=True` 成图；custom 相对图模式仍快 `1.39x-6.80x`。
- `AutoCorrFusedAggregate`、`S6ScanFused`、`TirexSlstmCell` 的原模型表达当前不能直接完整成图，原因分别是缺少 `aten.roll.default`、`aten.softplus.default`、`aten.log_sigmoid_forward.default` converter。
- 为避免使用较弱的 eager 路径作为唯一基线，测试将上述 3 个表达人工改写为数学等价、TorchAir 可转换的基础算子组合。custom 与这种改写后的强图基线相比仍快 `7.39x`、`6.48x`、`1.78x`。
- 因此这 3 组结果不能写成“原模型直接 TorchAir 成图”。框架路径需要额外改写模型表达，自定义算子则直接实现目标语义，接入时只替换为 ACLNN 调用。
- Koopa 不属于图优化问题：底层 `torch.linalg.lstsq` 本身不支持 NPU，TorchAir 无法把 CPU fallback 自动转换为 NPU 图。

## 算子实现逻辑

第一批算子主要采用三类实现方式：

1. **时间扫描融合**：`SelectiveScan1D`、`S6ScanFused`、CfC/coRNN/SRU/UnICORNN/LTC 将时间步循环放入单个 Ascend C kernel，状态在 kernel 内持续更新，避免每个时间步重新 launch 一组 NPU 算子。
2. **lag 聚合融合**：`AutoCorrFusedAggregate` 在一个 kernel 中完成相关性筛选、lag 权重计算和时移聚合，避免框架侧反复执行 `roll/gather/reduce`。
3. **不支持路径替换**：`BatchSpdInvFp32` 利用批量 SPD 矩阵求逆替换 Koopa/DMD 中会回落 CPU 的最小二乘关键路径；`TirexSlstmCell` 将 sLSTM 的门控、稳定化和状态更新合并执行。

每个算子的公式、输入输出、tiling、GM/UB 数据流和误差说明均记录在对应的 `docs/algorithm.md` 与 `docs/api_reference.md` 中。

## 交付件完整度

10 个正式算子均提供以下核心交付内容：

- `CMakeLists.txt`、`README.md`
- `op_host/`、`op_kernel/`
- `docs/algorithm.md`、`docs/api_reference.md`
- `docs/benchmark.md`、`docs/test_report.md`
- `examples/`、`tests/`
- 按算子构建方式提供的 msopgen 配置、构建脚本或直接 CMake/host API 接入文件

项目级补充交付件包括：

- TorchAir 第一批统一报告和原始结果 JSON
- 第一批 10 个算子合入评审材料与逐页讲稿（Markdown）
- CANN 算子开发规则、SCA 排查经验和 LLM 研发提示词模板

## 验证情况

- 10 个算子已在 Ascend 910B3 完成 CMake/msopgen 构建、ACLNN runtime、正确性和 benchmark 验证。
- 9 个可由 torch_npu 表达的算子完成 eager NPU 对照；其中 6 个完成原表达直接 TorchAir fullgraph，3 个完成 converter 缺失确认及等价改写后的 TorchAir 强基线。
- Koopa 完成 CPU fallback、DMD 子图和完整模型推理对照，证明自定义算子消除了 NPU 不支持路径。
- 本地完成 Python 语法、JSON 解析、选择性扫描单测及 `git diff --check`。

本地检查命令：

```bash
python -m compileall -q prediction/ProcessControl/TimeSeriesForecast
python -m pytest prediction/ProcessControl/TimeSeriesForecast/selective_scan_1d/tests/test_selective_scan_1d.py -q
git diff --check
```

CANN 环境构建示例：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd prediction/ProcessControl/TimeSeriesForecast/<op>
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
```

带 msopgen 的算子可按各目录 `README.md` 或 `docs/api_reference.md` 重新生成 ACLNN 包，并运行 `tests/` 下的 probe、模型适配或 benchmark 脚本。

## 结论

本 PR 的价值不只是“把 CPU 代码移到 NPU”。其中 9 个算子是在同一 NPU 上，相对 eager torch_npu 或 TorchAir 图模式减少框架拆分和调度开销；Koopa 则补齐了框架当前不支持、必须回落 CPU 的关键计算。第一批 10 个算子均具有明确模型入口、可复现性能数据、正确性验证和完整交付目录，可作为后续时序模型 Ascend 适配与算子扩展的基础。

## Checklist

- [x] 仅提交第一批 10 个正式算子及其项目级交付文档。
- [x] 目录结构按 `PIDModelFit/CANN_OPERATOR_RULES.md` 整理。
- [x] 每个算子包含 kernel、host、docs、examples 和 tests。
- [x] 性能数据区分 eager NPU、TorchAir 强基线和 Koopa CPU fallback。
- [x] TorchAir 结果区分原表达直接成图与等价改写后成图。
- [x] 在 Ascend 910B3 完成构建、运行时、正确性和性能复验。
- [x] 作者邮箱为 `aliutec@163.com`。
