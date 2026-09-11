# TirexSlstmCell sLSTM 单元融合算子

`TirexSlstmCell` 面向 TiRex/xLSTM 时序基础模型，将一个完整序列上的稳定化 sLSTM cell 递推融合为单个 Ascend C 算子。算子覆盖门控计算、循环状态更新和整个时间序列输出，适用于在昇腾 NPU 上部署 TiRex 等非 Transformer 时序模型。

## 应用场景

TiRex 的核心时序模块包含 sLSTM 递归。常规实现沿序列逐时间步调用多个点运算，分别处理 sigmoid、tanh、log-sigmoid、状态归一化和写回，导致 kernel launch 次数多、临时 Tensor 多，难以充分利用 NPU 计算资源。

`TirexSlstmCell` 将 sLSTM 的时间递推下沉到设备端，在一个算子中完成循环状态维护和输出生成。当前软件栈缺少 `aten.log_sigmoid_forward.default` converter 时，算子仍可直接表达目标 sLSTM 语义，不需要修改模型为人工等价表达。

## 算子功能

算子对每个 batch/head 独立执行序列扫描：

- 将输入投影、循环投影和 bias 合成为四组门控的 pre-activation；
- 按 `[i, f, z, o]` 顺序计算 input、forget、cell candidate 和 output gates；
- 使用稳定化的 log-domain 计算处理 sLSTM 的归一化状态；
- 递推更新 `h`、`c`、`n`、`m` 四类状态；
- 输出每个时间步的 hidden state，并返回最终状态；
- 将独立的 `(batch, head)` 扫描组分配到 AI Vector Core 并行执行。

## 接口

逻辑接口为：

```text
TirexSlstmCell(input, recurrent_kernel, bias, initial_state)
    -> output, final_state
```

构建后通过生成的 `aclnnTirexSlstmCellGetWorkspaceSize` 和
`aclnnTirexSlstmCell` 接口完成 workspace 查询与执行；具体 C API 参数顺序以生成的头文件为准。

| 张量 | Shape | dtype | format | 说明 |
|---|---|---|---|---|
| `input` | `[B, S, 4*H]` | `float32` | `ND` | 输入投影，末维门控顺序为 `[i, f, z, o]` |
| `recurrent_kernel` | `[num_heads, Hd, 4*Hd]` | `float32` | `ND` | 每个 head 的循环投影矩阵，`H=num_heads*Hd` |
| `bias` | `[4*H]` | `float32` | `ND` | 门控 bias，顺序为 `[i, f, z, o]` |
| `initial_state` | `[4, B, H]` | `float32` | `ND` | 初始状态，顺序为 `[h, c, n, m]` |
| `output` | `[B, S, H]` | `float32` | `ND` | 每个时间步的 hidden 输出 |
| `final_state` | `[4, B, H]` | `float32` | `ND` | 最终状态，顺序为 `[h, c, n, m]` |

典型验证规模为 `B=64,S=64,H=512,num_heads=4,Hd=128`。详细 shape 约束和调用参数见 [docs/api_reference.md](docs/api_reference.md)。

## 算法概要

对每个 head 和时间步，算子按以下流程更新状态：

1. 计算输入投影、循环投影和 bias 的和，得到四组 gate pre-activation；
2. 通过稳定的 log-sigmoid 和指数运算计算输入门与遗忘门；
3. 计算 `tanh` candidate 和 sigmoid output gate；
4. 在 log-domain 稳定化系数下更新 `c`、`n` 和 `m`；
5. 计算并写回 `h`，同时写入对应时间步的 `output`；
6. 扫描结束后输出 `[h,c,n,m]` 的 `final_state`。

完整递推公式、状态布局、并行化方式和同步策略见 [docs/algorithm.md](docs/algorithm.md)。

## 精度与性能

验证环境为 Ascend 910B3，主测试规模为 `B=64,S=64,H=512,num_heads=4`。baseline 与自定义算子均在 NPU 上执行：

| 场景 | torch_npu baseline | 自定义算子 | 加速比 | 精度 |
|---|---:|---:|---:|---:|
| sLSTM cell | `52.349 ms` | `3.658 ms` | `14.31x` | max diff `1.45e-06` |
| sLSTM layer | `53.910 ms` | `5.153 ms` | `10.46x` | 与 cell 输出路径一致 |
| TorchAir 补充测试 | `6.318 ms` | `3.542 ms` | `1.78x` | max diff `< 7.45e-08` |

原始模型表达在 TorchAir 中因缺少 `log_sigmoid` converter 无法直接完整成图；TorchAir 补充结果来自人工等价改写后的表达，仅用于说明可比较的设备侧性能，不代表 TorchAir 可以自动转换原始模型。详细环境、测试规模和计时口径见 [docs/benchmark.md](docs/benchmark.md)，测试结论见 [docs/test_report.md](docs/test_report.md)。

## 构建与测试

在已配置 CANN 环境的 Linux/Ascend 910B3 环境中，可按以下方式独立构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

Python ctypes probe、正确性验证和 TiRex 风格 E2E 脚本位于 `tests/`。具体测试前置条件和命令见 [docs/test_report.md](docs/test_report.md)。

## 目录结构

- `op_host/`：Host 参数处理、shape 推导、tiling 和算子注册；
- `op_kernel/`：Ascend C sLSTM cell 融合 kernel；
- `msopgen/`：算子 IR 定义；
- `tests/`：ctypes probe、正确性和模型级 E2E 验证；
- `docs/`：算法、API、性能和测试报告。

## 相关文档

- [算法说明](docs/algorithm.md)
- [API 参考](docs/api_reference.md)
- [性能报告](docs/benchmark.md)
- [测试报告](docs/test_report.md)
