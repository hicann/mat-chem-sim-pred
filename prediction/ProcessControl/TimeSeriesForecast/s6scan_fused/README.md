# S6scanFused 选择性状态空间扫描融合算子

`S6scanFused` 面向 S6/Mamba 风格的状态空间模型（State Space Model，SSM），将时间维上的选择性状态递推、输入相关离散化和输出计算融合为一个 Ascend C 算子。它适用于需要在 NPU 上高效执行长序列递推的时序预测模型。

## 应用场景

S6 是 Mamba 类模型的核心递推结构。传统 PyTorch/torch_npu 实现通常在时间维逐步执行状态更新，并拆分为 softplus、指数衰减、门控和逐元素运算，产生大量中间 Tensor 和 kernel launch。`S6scanFused` 将这些操作合并到单个 kernel 中，减少 launch 和中间结果物化开销。

算子直接表达 S6 门控递推语义；在当前软件栈缺少 `aten.softplus.default` converter 时，不需要修改模型为人工等价的基础张量表达。

## 算子功能

给定输入序列、扫描权重和 bias，算子沿时间维 `L` 执行选择性状态扫描：

- 对每个 batch 和时间步计算输入相关的门控/步长参数；
- 使用 softplus 构造稳定的步长；
- 计算状态衰减和输入注入；
- 递推更新隐状态并输出整个 hidden 序列；
- 在 Vector Core 上完成批量和序列计算，减少逐时间步的主机调度。

核心递推公式、权重打包方式和边界条件以 [docs/algorithm.md](docs/algorithm.md) 为准。

## 接口

逻辑接口为：

```text
S6scanFused(x, weight, bias) -> output
```

构建后通过生成的 `aclnnS6scanFusedGetWorkspaceSize` 和
`aclnnS6scanFused` 接口完成 workspace 查询与执行；具体 C API 参数顺序以生成的头文件为准。

| 张量 | Shape | dtype | format | 说明 |
|---|---|---|---|---|
| `x` | `[B, L, IN]` | `float32` | `ND` | 输入时序特征 |
| `weight` | `[3*IN, H]` | `float32` | `ND` | 按 `Wd、Wb、Wx` 顺序拼接的扫描权重 |
| `bias` | `[4*H]` | `float32` | `ND` | 按 `bd、bb、bx、a_param` 顺序拼接的 bias |
| `output` | `[B, L, H]` | `float32` | `ND` | 每个时间步的 hidden 输出 |

具体 shape 约束、参数打包方式和生成的 API 签名见 [docs/api_reference.md](docs/api_reference.md)。

## 算法概要

对输入序列的每个时间步，算子完成：

1. 根据输入和权重计算选择性扫描参数；
2. 对步长执行稳定的 softplus 计算；
3. 计算状态转移的指数衰减项和输入门控项；
4. 更新 SSM 隐状态；
5. 将当前隐状态写入 `[B, L, H]` 输出序列。

算子以时间步递推为计算主线，并在设备端融合逐元素非线性、状态更新和写回操作。完整公式、并行策略和数值处理见 [docs/algorithm.md](docs/algorithm.md)。

## 精度与性能

验证环境为 Ascend 910B3，数据类型为 FP32，baseline 为 torch_npu 的逐时间步 loop 实现。代表性结果如下：

| 场景 | baseline | 自定义算子 | 加速比 | 精度 |
|---|---:|---:|---:|---:|
| 单层 component | torch_npu 逐步递推 | `S6scanFused` | `136.56x` | 见测试报告 |
| 3-layer encoder E2E | torch_npu 框架路径 | `S6scanFused` | `44.15x` | 见测试报告 |
| `B=32,L=336,IN=11,H=64` TorchAir 补充测试 | `4.372 ms` | `0.674 ms` | `6.48x` | max diff `< 4.47e-08` |

原始 S6 表达在 TorchAir 中因缺少 `softplus` converter 无法直接完整成图；上表中的 TorchAir 结果来自人工等价改写 softplus 后的补充基线，不代表 TorchAir 可以自动转换原始表达。详细数据见 [docs/benchmark.md](docs/benchmark.md) 和 [docs/test_report.md](docs/test_report.md)。

## 构建与测试

在已配置 CANN 环境的 Linux/Ascend 910B3 环境中，可按以下方式独立构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

Python ctypes probe 和模型级 E2E 验证位于 `tests/`。当前交付以 Python probe 为主，具体测试前置条件和命令见 [docs/test_report.md](docs/test_report.md)。

## 目录结构

- `op_host/`：Host 参数处理、shape 推导、tiling 和算子注册；
- `op_kernel/`：Ascend C S6 选择性扫描融合 kernel；
- `msopgen/`：算子 IR 定义；
- `tests/`：ctypes probe、正确性和模型级 E2E 验证；
- `docs/`：算法、API、性能和测试报告。

## 相关文档

- [算法说明](docs/algorithm.md)
- [API 参考](docs/api_reference.md)
- [性能报告](docs/benchmark.md)
- [测试报告](docs/test_report.md)
