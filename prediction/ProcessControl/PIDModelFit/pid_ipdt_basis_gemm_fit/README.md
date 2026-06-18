# PidIpdtBasisGemmFit

## 功能说明

`PidIpdtBasisGemmFit` 面向 IPDT（Integrator Plus Dead Time，积分加纯滞后）模型辨识。它用于液位、库存、物料累积等积分过程的候选模型筛选。

## IPDT 模型

```text
G(s) = K * exp(-L s) / s
```

候选网格主要由纯滞后 `L` 和离散积分响应形态构成，增益 `K` 通过最小二乘解析估计。

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
cd prediction/ProcessControl/PIDModelFit/pid_ipdt_basis_gemm_fit
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
make -j$(nproc)
```

## 测试

```bash
python tests/test_pid_ipdt_basis_gemm_fit.py
python tests/benchmark_pid_ipdt_basis_gemm_fit.py
```

NPU smoke 用例见 `examples/test_aclnn_pid_ipdt_basis_gemm_fit.cpp`。

## 性能记录

Ascend 910B1 原型验证：

| 配置 | CPU 多线程完整 fit | NPU cached 端到端 | NPU cold 端到端 |
|------|--------------------|-------------------|-----------------|
| `B=64,N=1024,M=256` | `9.64 ms` | `0.40 ms` | `0.63 ms` |
| `B=128,N=1024,M=512` | `32.63 ms` | `0.64 ms` | `1.30 ms` |
