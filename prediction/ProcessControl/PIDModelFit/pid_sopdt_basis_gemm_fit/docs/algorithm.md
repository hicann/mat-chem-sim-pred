# PidSopdtBasisGemmFit 算法说明

## 原理

SOPDT 使用两个一阶惯性环节串联描述对象动态。候选基函数由 `(T1, T2, L)` 网格生成，单位增益响应记为 `g_m(t_i)`。固定候选后，最优增益和 SSE 仍为：

```text
K_hat[b, m] = dot[b, m] / basis_norm[m]
SSE[b, m] = y_energy[b] - dot[b, m]^2 / basis_norm[m]
```

SOPDT 的候选维度通常高于 FOPDT/IPDT，因此将候选扫描矩阵化后，NPU 的收益更稳定。

## NPU 实现

- `basis_t[N, M]` 来自 SOPDT 的 `(T1,T2,L)` 候选网格。
- CANN MatMul 负责 `B x N` 与 `N x M` 的主计算。
- 自定义 Ascend C 算子融合计算 `best_sse/best_k/best_idx`，避免输出完整评分矩阵。

## 精度

测试构造带真实 SOPDT 候选的数据，断言最优候选下标、增益和 SSE 与 NumPy 参考一致。
