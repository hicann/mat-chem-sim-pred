# PidFopdtBasisGemmFit

## 功能说明

`PidFopdtBasisGemmFit` 面向 FOPDT（First Order Plus Dead Time，一阶惯性加纯滞后）模型辨识。算子接收 CANN MatMul 生成的 `dot[B, M]`，融合完成增益估计、SSE 评分和最优候选选择。

## FOPDT 模型

```text
G(s) = K * exp(-L s) / (T s + 1)
```

其中 `K` 为过程增益，`T` 为一阶时间常数，`L` 为纯滞后。候选网格由 `(T, L)` 构成，`K` 在 reduce 阶段解析估计。

## 输入输出

| 名称 | 类型 | Shape | 说明 |
|------|------|-------|------|
| `dot` | float32 | `[B, M]` | `y_centered @ basis_t` 的结果 |
| `basis_norm` | float32 | `[M]` | 每个候选基函数的平方和 |
| `y_energy` | float32 | `[B]` | 每条回路中心化输出平方和 |
| `best_sse` | float32 | `[B]` | 每条回路最优 SSE |
| `best_k` | float32 | `[B]` | 每条回路估计增益 |
| `best_idx` | int32 | `[B]` | 每条回路最优候选下标 |

## 输入从哪里来

该算子的 API 站在模型辨识预处理之后，并不直接接收原始 `pv/sv/mv`。原始数据通常先被处理成：

```text
pv/mv 历史窗口
  -> 去基线/中心化得到 y_centered[B, N]
  -> 枚举候选 (T,L)，按 mv 输入轨迹生成单位增益响应 basis_t[N, M]
  -> CANN MatMul 得到 dot[B, M] = y_centered @ basis_t
  -> basis_norm[M] = sum_n basis_t[n,m]^2
  -> y_energy[B] = sum_n y_centered[b,n]^2
```

其中 `basis_t` 不是采集来的，而是“假设某个候选 FOPDT 模型是真的且 `K=1`，面对这段 `mv` 输入会产生的 `pv` 响应形状”。`sv` 通常用于判断工况、设定值变化或后续控制性能评估；basis-GEMM fit 的核心是 `mv -> pv` 的过程响应。

## 计算方法与小例子

对每个候选 `m`：

```text
K_hat[b,m] = dot[b,m] / basis_norm[m]
SSE[b,m] = y_energy[b] - dot[b,m]^2 / basis_norm[m]
best[b] = argmin_m SSE[b,m]
```

例如 1 条回路、3 个采样点、2 个候选：

```text
y_centered = [1, 2, 3]
basis_0 = [1, 1, 1]
basis_1 = [0, 1, 2]
```

则：

```text
dot = [6, 8]
basis_norm = [3, 5]
y_energy = 14
```

候选 0：

```text
K_hat = 6 / 3 = 2
SSE = 14 - 6^2 / 3 = 2
```

候选 1：

```text
K_hat = 8 / 5 = 1.6
SSE = 14 - 8^2 / 5 = 1.2
```

所以候选 1 胜出，输出 `best_idx=1`、`best_k=1.6`、`best_sse=1.2`。

## 构建

```bash
cd prediction/ProcessControl/PIDModelFit/pid_fopdt_basis_gemm_fit
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
make -j$(nproc)
```

## 测试

```bash
python tests/test_pid_fopdt_basis_gemm_fit.py
python tests/benchmark_pid_fopdt_basis_gemm_fit.py
```

NPU smoke 用例见 `examples/test_aclnn_pid_fopdt_basis_gemm_fit.cpp`。

NPU/CPU reduce-only benchmark：

```bash
./build/benchmark_pid_fopdt_basis_gemm_fit 3 4096 1024 10 2 64
```

NPU/CPU `MatMul + reduce` pipeline benchmark：

```bash
./build/benchmark_pid_fopdt_basis_gemm_pipeline 3 64 1024 256 5 2 64
```

## 性能记录

Ascend 910B3 原型验证：

| 配置 | CPU 多线程完整 fit | NPU cached 端到端 | NPU cold 端到端 |
|------|--------------------|-------------------|-----------------|
| `B=64,N=1024,M=256` | `9.22 ms` | `0.43 ms` | `0.87 ms` |
| `B=128,N=1024,M=512` | `31.55 ms` | `0.99 ms` | `1.02 ms` |

详细性能与精度数据见 [测试报告](docs/benchmark.md)。
