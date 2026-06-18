# PidSopdtBasisGemmFit

## 功能说明

`PidSopdtBasisGemmFit` 面向 SOPDT（Second Order Plus Dead Time，二阶惯性加纯滞后）模型辨识。SOPDT 候选通常覆盖两个时间常数和纯滞后，候选空间更大，更适合用 NPU basis-GEMM 路线加速。

## SOPDT 模型

```text
G(s) = K * exp(-L s) / ((T1 s + 1)(T2 s + 1))
```

其中 `T1`、`T2` 为两个主导时间常数，`L` 为纯滞后，`K` 在 reduce 阶段解析估计。

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
cd prediction/ProcessControl/PIDModelFit/pid_sopdt_basis_gemm_fit
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
make -j$(nproc)
```

## 测试

```bash
python tests/test_pid_sopdt_basis_gemm_fit.py
python tests/benchmark_pid_sopdt_basis_gemm_fit.py
```

NPU smoke 用例见 `examples/test_aclnn_pid_sopdt_basis_gemm_fit.cpp`。

## 性能记录

Ascend 910B1 原型验证：

| 配置 | CPU 多线程完整 fit | NPU cached 端到端 | NPU cold 端到端 |
|------|--------------------|-------------------|-----------------|
| `B=64,N=1024,M=256` | `8.46 ms` | `0.57 ms` | `0.87 ms` |
| `B=128,N=1024,M=512` | `33.40 ms` | `0.87 ms` | `1.19 ms` |
