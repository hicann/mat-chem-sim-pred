# CfcScanFused API 说明

## 算子入口

```text
CfcScanFused(x, weight, bias) -> output
```

该算子保持原交付物的 msopgen/aclnn 形态。`msopgen/cfc_scan_fused_msopgen.json` 定义输入输出，构建后生成 `aclnn_cfc_scan_fused.h` 及 op_api。

## 输入输出

| 名称 | dtype | Shape | 说明 |
|------|-------|-------|------|
| `x` | float32 | `[B, L, IN]` | 输入序列 |
| `weight` | float32 | `[H + IN, 3H]` | transposed packed projection，`Wf1/Wf2/Wt` |
| `bias` | float32 | `[3H]` | `bf1/bf2/bt` |
| `output` | float32 | `[B, L, H]` | CfC hidden 序列 |

## Tiling

```text
batch     = x.dim(0)
length    = x.dim(1)
in_size   = x.dim(2)
hidden    = weight.dim(1) / 3
block_dim = min(batch, 32)
```

## 构建方式

主构建路径为 msopgen：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/cfc_scan_fused_msopgen.json -f aclnn -c ai_core-ascend910b -out build/msopgen_cfc_scan_fused -lan cpp
```

生成工程后接入本目录的 `op_kernel/cfc_scan_fused_kernel.cpp`、`op_host/cfc_scan_fused_host.cpp` 和 `op_host/cfc_scan_fused_tiling.h`。

## 约束

- 当前实现只支持 float32 ND。
- `weight.dim(1)` 必须能被 3 整除。
- `IN` 可以不是 8 的倍数；kernel 通过 `DataCopyPad` 处理非对齐输入。
