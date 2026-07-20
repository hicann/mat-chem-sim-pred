# TirexSlstmCell API 说明

## msopgen

msopgen/tirex_slstm_cell_msopgen.json

## 输入

| 名称 | dtype | format |
|------|-------|--------|
| input | float32 | ND |
| recurrent_kernel | float32 | ND |
| bias | float32 | ND |
| initial_state | float32 | ND |

## 输出

| 名称 | dtype | format |
|------|-------|--------|
| output | float32 | ND |
| final_state | float32 | ND |

具体 shape 约束以 docs/algorithm.md、源 probe 和 E2E 脚本为准。
