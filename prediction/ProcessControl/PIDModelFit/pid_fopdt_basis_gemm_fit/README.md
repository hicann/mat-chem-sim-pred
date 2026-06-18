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

## 构建

```bash
cd prediction/ProcessControl/PIDModelFit/pid_fopdt_basis_gemm_fit
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
make -j$(nproc)
```

## 测试

```bash
python tests/test_pid_fopdt_basis_gemm_fit.py
python tests/benchmark_pid_fopdt_basis_gemm_fit.py
```

NPU smoke 用例见 `examples/test_aclnn_pid_fopdt_basis_gemm_fit.cpp`。

## 性能记录

Ascend 910B1 原型验证：

| 配置 | CPU 多线程完整 fit | NPU cached 端到端 | NPU cold 端到端 |
|------|--------------------|-------------------|-----------------|
| `B=64,N=1024,M=256` | `9.22 ms` | `0.43 ms` | `0.87 ms` |
| `B=128,N=1024,M=512` | `31.55 ms` | `0.99 ms` | `1.02 ms` |
