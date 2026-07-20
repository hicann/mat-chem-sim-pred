# AutoCorrFusedAggregate API 说明

## 算子入口

```text
AutoCorrFusedAggregate(query, key, value, top_k=1) -> output
```

该算子保持原交付物的 msopgen/aclnn 形态。`msopgen/autocorr_fused_aggregate_msopgen.json` 定义输入输出和 `top_k` 属性，构建后生成 `aclnn_auto_corr_fused_aggregate.h` 及 op_api。

## 输入输出

| 名称 | dtype | Shape | 说明 |
|------|-------|-------|------|
| `query` | float32 | `[B, H, E, L]` | AutoCorrelation query |
| `key` | float32 | `[B, H, E, L]` | AutoCorrelation key |
| `value` | float32 | `[B, H, E, L]` | 被 lag 聚合的 value |
| `top_k` | int attr | scalar | 选取的 lag 数，默认 1，kernel 内部上限 32 |
| `output` | float32 | `[B, H, E, L]` | 聚合后的输出 |

## Tiling

```text
batch  = query.dim(0)
heads  = query.dim(1)
embed  = query.dim(2)
length = query.dim(3)
top_k  = attr.top_k
block_dim = 32
```

## 构建方式

主构建路径为 msopgen：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/autocorr_fused_aggregate_msopgen.json -f aclnn -c ai_core-ascend910b -out build/msopgen_autocorr_fused_aggregate -lan cpp
```

生成工程后接入本目录的 `op_kernel/autocorr_fused_aggregate_kernel.cpp`、`op_host/autocorr_fused_aggregate_host.cpp` 和 `op_host/auto_corr_fused_aggregate_tiling.h`。
