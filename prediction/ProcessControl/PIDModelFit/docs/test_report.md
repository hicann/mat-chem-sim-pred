# PID 模型辨识算子组测试报告

## 测试对象

| 算子 | 目录 | Host API |
|------|------|----------|
| FOPDT basis-GEMM fit | `pid_fopdt_basis_gemm_fit/` | `aclnnPidFopdtBasisGemmFit` |
| IPDT basis-GEMM fit | `pid_ipdt_basis_gemm_fit/` | `aclnnPidIpdtBasisGemmFit` |
| SOPDT basis-GEMM fit | `pid_sopdt_basis_gemm_fit/` | `aclnnPidSopdtBasisGemmFit` |

三个算子均使用 `common/` 中的 basis-GEMM best reduce 底座，但保持独立目录、独立入口、独立构建和独立测试。

## 测试环境

| 项目 | 环境 |
|------|------|
| 本地开发机 | Windows / Python 3.11 |
| NPU 测试机 | node202 |
| NPU | Ascend 910B1 |
| CANN | node202 `/usr/local/Ascend/ascend-toolkit` |
| 构建工具 | CMake + Ascend C CMake |

## 测试结论

- 本地 NumPy 参考精度测试通过。
- 三个独立算子在 node202 上均完成 CANN 编译。
- 三个独立算子均完成 ACL smoke，输出 `best_sse/best_k/best_idx` 与 CPU 预期一致。
- 原型性能验证显示，在多回路、多候选典型规模下，NPU cached 端到端路径显著快于 CPU 多线程完整 fit。

## 1. 本地 Python 精度测试

### 命令

```bash
python -m pytest \
  prediction/ProcessControl/PIDModelFit/pid_fopdt_basis_gemm_fit/tests/test_pid_fopdt_basis_gemm_fit.py \
  prediction/ProcessControl/PIDModelFit/pid_ipdt_basis_gemm_fit/tests/test_pid_ipdt_basis_gemm_fit.py \
  prediction/ProcessControl/PIDModelFit/pid_sopdt_basis_gemm_fit/tests/test_pid_sopdt_basis_gemm_fit.py \
  -q
```

### 结果

```text
6 passed
```

### 覆盖内容

| 测试项 | 覆盖内容 |
|--------|----------|
| FOPDT best reduce | `dot -> best_sse/best_k/best_idx` 与 NumPy 参考一致 |
| IPDT best reduce | 支持积分对象候选，处理等价最优候选场景 |
| SOPDT best reduce | 支持二阶对象候选，验证高候选维度路径 |
| 增益估计 | `best_k` 接近构造数据的真实过程增益 |
| 下标合法性 | `best_idx` 在候选范围内 |
| 数值稳定性 | SSE 浮点残差在阈值内 |

## 2. Python 语法检查

### 命令

```bash
python -m compileall -q prediction/ProcessControl/PIDModelFit
```

### 结果

通过，无语法错误。

## 3. Ascend C 编译测试

### 命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

for op in pid_fopdt_basis_gemm_fit pid_ipdt_basis_gemm_fit pid_sopdt_basis_gemm_fit; do
  cd prediction/ProcessControl/PIDModelFit/$op
  rm -rf build
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
  cmake --build build -j 2
done
```

### 结果

| 算子 | CMake 配置 | Ascend C kernel 编译 | Host lib 编译 | Example 编译 |
|------|------------|----------------------|---------------|--------------|
| FOPDT | 通过 | 通过 | 通过 | 通过 |
| IPDT | 通过 | 通过 | 通过 | 通过 |
| SOPDT | 通过 | 通过 | 通过 | 通过 |

生成目标包括：

- `libpid_fopdt_basis_gemm_fit_kernel_lib.so`
- `libpid_fopdt_basis_gemm_fit_host.so`
- `test_aclnn_pid_fopdt_basis_gemm_fit`
- `libpid_ipdt_basis_gemm_fit_kernel_lib.so`
- `libpid_ipdt_basis_gemm_fit_host.so`
- `test_aclnn_pid_ipdt_basis_gemm_fit`
- `libpid_sopdt_basis_gemm_fit_kernel_lib.so`
- `libpid_sopdt_basis_gemm_fit_host.so`
- `test_aclnn_pid_sopdt_basis_gemm_fit`

## 4. NPU smoke 测试

### 测试数据

smoke 用例构造小规模输入：

```text
B = 2, M = 4
dot = [[1, 3, 6, 2],
       [1, 4, 2, 8]]
basis_norm = [1, 2, 4, 8]
y_energy = [10, 20]
```

CPU 预期结果：

```text
best_sse = [1, 12]
best_k   = [1.5, 2]
best_idx = [2, 1]
```

### 结果

```text
PidFopdtBasisGemmFit smoke best_sse=[1, 12] best_k=[1.5, 2] best_idx=[2, 1]
PASSED

PidIpdtBasisGemmFit smoke best_sse=[1, 12] best_k=[1.5, 2] best_idx=[2, 1]
PASSED

PidSopdtBasisGemmFit smoke best_sse=[1, 12] best_k=[1.5, 2] best_idx=[2, 1]
PASSED
```

## 5. 性能验证记录

以下数据来自 Ascend 910B1 原型验证，路径为 basis-GEMM：CANN MatMul 完成密集计算，自定义算子完成 best reduce。

| 模型 | 配置 | CPU 单线程完整 fit | CPU 多线程完整 fit | NPU cached 端到端 | NPU cold 端到端 | NPU cold + host basis build |
|------|------|--------------------|--------------------|-------------------|-----------------|-----------------------------|
| FOPDT | `B=64,N=1024,M=256` | `98.14 ms` | `9.22 ms` | `0.43 ms` | `0.87 ms` | `3.29 ms` |
| IPDT | `B=64,N=1024,M=256` | `98.21 ms` | `9.64 ms` | `0.40 ms` | `0.63 ms` | `3.09 ms` |
| SOPDT | `B=64,N=1024,M=256` | `96.00 ms` | `8.46 ms` | `0.57 ms` | `0.87 ms` | `3.35 ms` |
| FOPDT | `B=128,N=1024,M=512` | `460.84 ms` | `31.55 ms` | `0.99 ms` | `1.02 ms` | `5.96 ms` |
| IPDT | `B=128,N=1024,M=512` | `461.22 ms` | `32.63 ms` | `0.64 ms` | `1.30 ms` | `6.20 ms` |
| SOPDT | `B=128,N=1024,M=512` | `471.36 ms` | `33.40 ms` | `0.87 ms` | `1.19 ms` | `6.14 ms` |

### 性能结论

- 在 `B=64,N=1024,M=256` 及以上规模，NPU cached 端到端路径相对 CPU 多线程完整 fit 有明显优势。
- `cold + host basis build` 仍快于 CPU 多线程完整 rebuild 路径，适合候选 basis 可缓存或半缓存的在线整定场景。
- SOPDT 候选空间更大，后续扩大 `(T1,T2,L)` 网格时更能体现矩阵化和 NPU 路线价值。

## 6. 已知说明

- IPDT 场景可能存在多个滞后候选获得相同最优 SSE，因此测试重点校验 `best_sse`、`best_k`、`best_idx` 合法性和 reduce 公式一致性，而不是强制唯一候选下标。
- 当前自定义算子负责 best reduce，GEMM 主计算复用 CANN 内置 MatMul。后续可继续做 UB staging、候选 tile 向量化和图融合优化。
