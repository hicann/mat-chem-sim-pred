# CornnScanFused API 说明

## 算子入口

```text
CornnScanFused(x, weight, bias) -> output
```

该算子保持原交付物的 msopgen/aclnn 形态。`msopgen/cornn_scan_fused_msopgen.json` 定义输入输出，构建后生成 `aclnn_cornn_scan_fused.h` 及 op_api。

## 输入输出

| 名称 | dtype | Shape | 说明 |
|------|-------|-------|------|
| `x` | float32 | `[B, L, IN]` | 输入序列 |
| `weight` | float32 | `[2H + IN, H]` | transposed packed projection，`Wy/Wz/V` |
| `bias` | float32 | `[H]` | projection bias |
| `output` | float32 | `[B, L, H]` | coRNN position state `y` 序列 |

## Tiling

```text
batch     = x.dim(0)
length    = x.dim(1)
in_size   = x.dim(2)
hidden    = weight.dim(1)
block_dim = min(batch, 32)
```

## 内置常量

```text
dt = 0.042
gamma = 1.0
eps = 1.0
```

这些常量同时写入 kernel、C++ CPU oracle 和 Python E2E harness。若后续调参，三处需要同步修改。

## 构建方式

主构建路径为 msopgen：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/cornn_scan_fused_msopgen.json -f aclnn -c ai_core-ascend910b -out build/msopgen_cornn_scan_fused -lan cpp
```

生成工程后接入本目录的 `op_kernel/cornn_scan_fused_kernel.cpp`、`op_host/cornn_scan_fused_host.cpp` 和 `op_host/cornn_scan_fused_tiling.h`。
