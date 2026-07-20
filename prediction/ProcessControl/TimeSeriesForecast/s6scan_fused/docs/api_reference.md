# S6scanFused API 说明

## 算子入口

`	ext
S6scanFused(x, weight, bias) -> output
`

该算子采用 msopgen/aclnn 形态。msopgen/s6scan_fused_msopgen.json 定义输入输出，构建后生成对应 clnnS6scanFused op_api。

## 输入输出

| 名称 | dtype | Shape | 说明 |
|------|-------|-------|------|
| x | float32 | [B, L, IN] | 输入序列 |
| weight | float32 | [*, H] | fused scan 权重，具体打包方式见 docs/algorithm.md |
| ias | float32 | [*] | fused scan bias，具体打包方式见 docs/algorithm.md |
| output | float32 | [B, L, H] | 输出 hidden 序列 |

## Tiling

`	ext
batch     = x.dim(0)
length    = x.dim(1)
in_size   = x.dim(2)
hidden    = weight.dim(1)
block_dim = min(batch, 32)
`

## 构建方式

`ash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/s6scan_fused_msopgen.json -f aclnn -c ai_core-ascend910b -out build/msopgen_s6scan_fused -lan cpp
`

生成工程后接入本目录的 op_kernel/s6scan_fused_kernel.cpp、op_host/s6scan_fused_host.cpp 和 op_host/s6scan_fused_tiling.h。
