# SruScanFused API 说明

## 算子入口

```text
SruScanFused(x, weight, bias) -> output
```

该算子保持原交付物的 msopgen/aclnn 形态。`msopgen/sru_scan_fused_msopgen.json` 定义输入输出，构建后生成 `aclnn_sru_scan_fused.h` 及 op_api。

## 输入输出

| 名称 | dtype | Shape | 说明 |
|------|-------|-------|------|
| `x` | float32 | `[B, L, IN]` | 输入序列 |
| `weight` | float32 | `[3IN, H]` | `W/Wf/Wr` column-packed transposed 权重 |
| `bias` | float32 | `[4H]` | `[v_f, v_r, b_f, b_r]` |
| `output` | float32 | `[B, L, H]` | SRU hidden 序列 |

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
- `weight.dim(0)` 应为 `3 * in_size`。
- `bias.dim(0)` 应为 `4 * hidden`。
