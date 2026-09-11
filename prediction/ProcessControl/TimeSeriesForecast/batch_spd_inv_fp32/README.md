# BatchSpdInvFp32 批量小型 SPD 矩阵求逆算子

`BatchSpdInvFp32` 面向 Koopa/DMD 等时序预测模型，为批量小型对称正定（Symmetric Positive Definite，SPD）矩阵提供 FP32 求逆能力。该算子将模型中无法在 NPU 上执行、会回退 CPU 的小矩阵求逆环节下沉到 Ascend C kernel，使后续 DMD 计算能够保持在设备侧完成。

## 应用场景

Koopa 在每个 KPLayer 中需要求解欠定线性系统。对于输入 `x` 和目标 `y`，其最小范数解可表示为：

```text
K = xᵀ (x xᵀ)⁻¹ y
```

其中 `G = x xᵀ` 是批量小型 SPD 矩阵。PyTorch 的 `torch.linalg.lstsq` 在目标 NPU 后端缺少对应实现时会回退 CPU，成为时序预测推理的主要瓶颈。本算子仅实现缺失的 `G⁻¹`，矩阵乘法仍使用 NPU 原生 `bmm`：

```text
G  = bmm(x, xᵀ)
Gi = BatchSpdInvFp32(G)
K  = bmm(xᵀ, bmm(Gi, y))
```

## 算子功能

算子对 batch 中每个 `m × m` SPD 矩阵独立求逆：

- 输入和输出均为 FP32、ND 格式；
- 支持批量矩阵处理，`m` 为小矩阵维度，Koopa 常用范围为 3～7，当前支持上限为 64；
- 使用 LDLᵀ 分解和三角矩阵求逆，避免平方根和 Cube/Matmul 开销；
- 在 Vector Core 上按 batch 分片并行计算；
- 通过 UB staging 完成 GM 数据搬运，避免多核场景下标量 GM 访问的缓存一致性问题；
- 对接近奇异或秩亏矩阵使用相对尺度的 pivot floor，避免结果产生 NaN/Inf。该处理是稳定性保护，不等价于严格的 Moore–Penrose 伪逆。

## 接口

逻辑接口为：

```text
BatchSpdInvFp32(g) -> gi
```

构建后通过生成的 `aclnnBatchSpdInvFp32GetWorkspaceSize` 和
`aclnnBatchSpdInvFp32` 接口完成 workspace 查询与执行；具体 C API 参数顺序以生成的头文件为准。

| 张量 | Shape | dtype | format | 说明 |
|---|---|---|---|---|
| `g` | `[B, m, m]` | `float32` | `ND` | 批量 SPD 输入矩阵，通常为 Gram 矩阵 `x xᵀ` |
| `gi` | `[B, m, m]` | `float32` | `ND` | 每个输入矩阵的逆矩阵 |

输入输出 shape 必须一致。`B > 0`，`1 ≤ m ≤ 64`；调用前应确保输入满足 SPD 或接近 SPD 的业务约束。

## 算法概要

对每个 batch 矩阵执行：

1. LDLᵀ 分解：`g = L D Lᵀ`；
2. 计算对角矩阵 `D⁻¹`；
3. 通过前代求解得到 `L⁻¹`；
4. 组装 `g⁻¹ = L⁻ᵀ D⁻¹ L⁻¹`；
5. 将结果写回对应 batch 的输出矩阵。

完整数学定义、tiling 和设备侧实现见 [docs/algorithm.md](docs/algorithm.md)，Host API 约束见 [docs/api_reference.md](docs/api_reference.md)。

## 精度与性能

验证环境为 Ascend 910B3。代表性结果如下：

| 场景 | 原框架路径 | 自定义算子路径 | 加速比 | 精度 |
|---|---:|---:|---:|---:|
| DMD，`B=16,m=3,E=128` | `lstsq` CPU fallback：`932.47 ms` | 全设备：`0.386 ms` | `2418.55x` | max diff `< 5.96e-08` |
| DMD，`B=32,m=7,E=128` | `lstsq` CPU fallback：`4341.75 ms` | 全设备：`0.339 ms` | `12823.56x` | max diff `< 6.80e-08` |
| Koopa E2E，`seq=336,pred=96,B=32` | `5300.64 ms` | `8.80 ms` | `602.19x` | max diff `< 2.38e-07` |

原框架路径包含 CPU fallback，与其他纯 NPU framework baseline 的性能口径不同。详细环境、测试规模和计时口径见 [docs/benchmark.md](docs/benchmark.md)，测试结论见 [docs/test_report.md](docs/test_report.md)。

## 构建与测试

在已配置 CANN 环境的 Linux/Ascend 910B3 环境中，可按以下方式独立构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

Python/ctypes 正确性验证、DMD 集成验证和 Koopa E2E 脚本位于 `tests/`，具体命令和前置条件见测试报告。

## 目录结构

- `op_host/`：Host 参数处理、shape 推导、tiling 和算子注册；
- `op_kernel/`：Ascend C LDLᵀ 求逆 kernel；
- `msopgen/`：算子 IR 定义；
- `tests/`：CPU reference、ctypes probe、正确性和 E2E 验证；
- `docs/`：算法、API、性能和测试报告。

## 相关文档

- [算法说明](docs/algorithm.md)
- [API 参考](docs/api_reference.md)
- [性能报告](docs/benchmark.md)
- [测试报告](docs/test_report.md)
