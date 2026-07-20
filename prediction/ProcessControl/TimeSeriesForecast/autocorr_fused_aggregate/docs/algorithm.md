# AutoCorrFusedAggregate 算法说明

## 背景

Autoformer 的 AutoCorrelation 模块会在序列维度上搜索周期性 lag，再按 top-k lag 对 `value` 做 circular shift 聚合。框架实现通常由多个小算子和 Python/torch 循环组成，`torch.roll`、`topk`、softmax、gather/reduce 的 launch 开销会在短中序列预测中成为热点。

## 计算定义

输入均为 float32 ND：

```text
query, key, value: [B, H, E, L]
top_k: int, 1 <= top_k <= 32
output: [B, H, E, L]
```

对每个 `(b, h, e)` 独立序列：

```text
score[lag] = sum_t query[t] * key[(t + lag) mod L]
lags = TopK(score, top_k)
weight = Softmax(score[lags])
output[t] = sum_i weight[i] * value[(t + lags[i]) mod L]
```

## 并行策略

kernel 将 `(B * H * E)` 个序列组分配给 AI Core。每个组内部顺序扫描 lag 和时间维，避免 framework path 中大量小算子调度和中间 Tensor 往返。`top_k` 在 tiling 中传入，kernel 内部上限为 32。

## 约束

- 当前实现支持 float32 ND。
- `query`、`key`、`value` 形状必须相同，rank 为 4。
- `top_k` 大于 `L` 时实际按 `L` 处理。
- 当前实现聚焦 Autoformer-style aggregation 热路径，不单独暴露 correlation map。
