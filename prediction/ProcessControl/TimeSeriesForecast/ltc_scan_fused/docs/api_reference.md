# LtcScanFused API 说明

## 算子入口

```text
LtcScanFused(x, weight, bias) -> output
```

该算子保持原交付物的 msopgen/aclnn 形态。`msopgen/ltc_scan_fused_msopgen.json` 定义输入输出，构建后生成 `aclnn_ltc_scan_fused.h` 及 op_api。

## 输入输出

| 名称 | dtype | Shape | 说明 |
|------|-------|-------|------|
| `x` | float32 | `[B, L, IN]` | 输入序列 |
| `weight` | float32 | `[H + IN, H]` | `[W_rec | W_in]` 的 column-packed transposed 权重 |
| `bias` | float32 | `[5H]` | `[b, gleak, Eleak, cm, A]` |
| `output` | float32 | `[B, L, H]` | LTC hidden 序列 |

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
- kernel 内常量 `dt=0.042, K=6` 与原 torch baseline 对齐。
- `weight.dim(0)` 应为 `hidden + in_size`。
- `bias.dim(0)` 应为 `5 * hidden`。
