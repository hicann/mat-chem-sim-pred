# UnicornnScanFused API 说明

## 算子入口

```text
UnicornnScanFused(x, weight, bias) -> output
```

该算子保持原交付物的 msopgen/aclnn 形态。`msopgen/unicornn_scan_fused_msopgen.json` 定义输入输出，构建后生成 `aclnn_unicornn_scan_fused.h` 及 op_api。

## 输入输出

| 名称 | dtype | Shape | 说明 |
|------|-------|-------|------|
| `x` | float32 | `[B, L, IN]` | 输入序列 |
| `weight` | float32 | `[IN + 2, H]` | `[V | w_diag | c]` 的 column-packed transposed 权重 |
| `bias` | float32 | `[H]` | UnICORNN bias |
| `output` | float32 | `[B, L, H]` | position state `y` 的完整序列 |

## Tiling

```text
batch     = x.dim(0)
length    = x.dim(1)
in_size   = x.dim(2)
hidden    = weight.dim(1)
block_dim = min(batch, 32)
```

## 约束

- 当前实现支持 float32 ND。
- kernel 内常量 `dt=0.042, alpha=1` 与原 torch baseline 对齐。
- `weight.dim(0)` 应为 `IN + 2`。
