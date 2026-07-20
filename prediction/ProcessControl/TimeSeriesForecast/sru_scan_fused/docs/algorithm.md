# SruScanFused 算法说明

## 背景

SRU(Simple Recurrent Unit) 将 gate 的主要计算放在 input projection 和 peephole 上，hidden-to-hidden 依赖很轻。框架实现通常仍然需要按 timestep 循环，导致大量小算子 launch；`SruScanFused` 将整段序列 recurrence 融合为一个 Ascend C kernel。

## 计算定义

```text
x:      [B, L, IN]
weight: [3IN, H]
bias:   [4H]
output: [B, L, H]
```

`weight` 为 column-packed transposed：

```text
[0:IN, :]      = W^T
[IN:2IN, :]    = Wf^T
[2IN:3IN, :]   = Wr^T
```

`bias = [v_f | v_r | b_f | b_r]`。

对每个 timestep：

```text
x_tilde = W x_s
f = sigmoid(Wf x_s + v_f * c + b_f)
r = sigmoid(Wr x_s + v_r * c + b_r)
c = f * c + (1 - f) * x_tilde
h = r * tanh(c) + (1 - r) * x_tilde
output[b, s, :] = h
```

## 并行策略

kernel 按 batch 切分 AI Core。每个 core 将权重和 bias 搬入 UB，在本 core 负责的 batch 分片内顺序扫描 `L`，cell state `c` 常驻 UB，避免每个 timestep 的 framework launch 和 GM state round-trip。

## 约束

- 当前实现支持 float32 ND。
- `weight.dim(0)` 必须等于 `3 * x.dim(2)`。
- `bias.dim(0)` 必须等于 `4 * weight.dim(1)`。
