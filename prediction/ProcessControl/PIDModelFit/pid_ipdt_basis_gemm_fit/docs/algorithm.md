# PidIpdtBasisGemmFit 算法说明

## 原理

IPDT 描述积分型对象，单位增益基函数是滞后后随时间累积的响应。对每个候选 `m`，仍可通过最小二乘解析估计增益：

```text
K_hat[b, m] = dot[b, m] / basis_norm[m]
SSE[b, m] = y_energy[b] - dot[b, m]^2 / basis_norm[m]
```

因此 IPDT 的 NPU 路径与 FOPDT 一样采用 basis-GEMM，但候选基函数生成逻辑不同。

## NPU 实现

- `basis_t[N, M]` 来自 IPDT 候选滞后网格。
- `dot[B, M]` 由 CANN MatMul 完成。
- 本算子只做 IPDT 语义下的 best reduce，输出 `best_sse/best_k/best_idx`。

## 精度

测试构造带真实 IPDT 候选的数据，断言最优候选下标、增益和 SSE 与 NumPy 参考一致。
