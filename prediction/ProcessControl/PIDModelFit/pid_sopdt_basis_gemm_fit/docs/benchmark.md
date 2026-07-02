# PidSopdtBasisGemmFit 测试报告

## 测试环境

- 设备：Ascend910B3，device 3
- 机器：node202
- 构建：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3`
- CPU 基线：64 线程完整 fit，包含 `dot = y_centered @ basis_t` 和 best reduce

## 测试命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd prediction/ProcessControl/PIDModelFit/pid_sopdt_basis_gemm_fit
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
export LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:${LD_LIBRARY_PATH:-}"

./build/test_aclnn_pid_sopdt_basis_gemm_fit 3
./build/benchmark_pid_sopdt_basis_gemm_pipeline 3 64 1024 256 5 2 64
```

## 正确性

smoke 已通过：

```text
PidSopdtBasisGemmFit smoke best_sse=[1, 12] best_k=[1.5, 2] best_idx=[2, 1]
PASSED
```

pipeline benchmark 与 CPU reference 对比：

```text
max_abs_sse=0.00378418
max_rel_sse=0.00378418
max_abs_k=1.54972e-06
idx_diff_count=0
```

## 性能结果

`B=64,N=1024,M=256`：

| 口径 | 耗时 | 对 CPU 64T 加速比 |
|------|------:|------:|
| CPU 64T 完整 fit | 10.3824 ms | 1.00x |
| NPU resident e2e | 0.166816 ms | 62.24x |
| NPU cold e2e | 0.406088 ms | 25.57x |

## 口径说明

- `resident e2e`：输入已在 Device，只统计 `aclnnMatmul + custom reduce + best result D2H`。
- `cold e2e`：统计输入 H2D、`aclnnMatmul + custom reduce` 和 best result D2H。
- `dot[B, M]` 常驻 Device，不回传 Host，直接作为 reduce 算子的输入。

## 结论

SOPDT 候选空间通常比 FOPDT/IPDT 更大，更适合矩阵化 `MatMul + reduce` pipeline。当前实测显示 SOPDT pipeline 在 resident 和 cold e2e 两种口径下均显著快于 CPU 64 线程完整 fit。
