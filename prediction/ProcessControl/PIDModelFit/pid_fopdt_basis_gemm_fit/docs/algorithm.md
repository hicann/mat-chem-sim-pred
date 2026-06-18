# PidFopdtBasisGemmFit 算法说明

## 原理

FOPDT 候选模型的单位增益响应记为 `g_m(t_i)`。对每条回路 `y_b(t_i)`，固定候选 `(T_m, L_m)` 后，最小二乘意义下的最优增益为：

```text
K_hat[b, m] = sum_i y_b(t_i) g_m(t_i) / sum_i g_m(t_i)^2
```

残差平方和为：

```text
SSE[b, m] = sum_i y_b(t_i)^2 - (sum_i y_b(t_i)g_m(t_i))^2 / sum_i g_m(t_i)^2
```

因此候选扫描可以拆成：

1. `dot = y_centered @ basis_t`
2. 自定义算子 reduce 得到 `best_sse/best_k/best_idx`

## NPU 实现

- MatMul 使用 CANN 内置 `aclnnMatmul`。
- 本算子按 16 条回路为一个任务单元切分 batch，分配到多个 AI Core。
- 每个 AI Core 对本回路范围扫描全部候选，融合计算 gain、SSE 与 argmin。
- 输出只保留每条回路的最优结果，避免输出完整 `SSE[B, M]`。

## 精度

Python NumPy 参考实现使用相同公式生成 `dot` 和 best reduce。测试断言：

- `best_idx` 命中构造数据的真实候选。
- `best_sse` 与 reduce 参考一致。
- `best_k` 接近构造过程增益。
