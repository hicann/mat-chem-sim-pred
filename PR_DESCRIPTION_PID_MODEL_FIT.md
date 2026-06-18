## 类型

- [x] 新功能
- [x] 构建过程或辅助工具的变动
- [x] 文档内容更新

## 描述

本 PR 在 `prediction/ProcessControl/PIDModelFit/` 下新增工业过程控制 PID 模型辨识算子组，面向 PID 整定、自整定和数字孪生仿真中的多回路、多候选模型筛选场景。

本次提交包含三个工程上独立的 Ascend C 算子：

| 算子 | 目录 | 公开接口 | 模型语义 |
|------|------|----------|----------|
| FOPDT basis-GEMM fit | `pid_fopdt_basis_gemm_fit/` | `aclnnPidFopdtBasisGemmFit` | 一阶惯性加纯滞后模型 |
| IPDT basis-GEMM fit | `pid_ipdt_basis_gemm_fit/` | `aclnnPidIpdtBasisGemmFit` | 积分加纯滞后模型 |
| SOPDT basis-GEMM fit | `pid_sopdt_basis_gemm_fit/` | `aclnnPidSopdtBasisGemmFit` | 二阶惯性加纯滞后模型 |

三个算子均具备独立的 `CMakeLists.txt`、`README.md`、`docs/algorithm.md`、`docs/api_reference.md`、`op_host/`、`op_kernel/`、`examples/`、`tests/`。公共目录 `common/` 只复用数学等价的 basis-GEMM best reduce 底座，避免把三类模型混成一个带 `family` 参数的算子。

### 算法路线

将传统逐回路、逐候选的模型辨识改写为：

```text
dot[B, M] = y_centered[B, N] x basis_t[N, M]
K_hat[b, m] = dot[b, m] / basis_norm[m]
SSE[b, m] = y_energy[b] - dot[b, m]^2 / basis_norm[m]
best_idx[b] = argmin_m SSE[b, m]
```

其中 GEMM 主计算复用 CANN 内置 MatMul，自定义算子融合完成 `K_hat/SSE/best_idx` reduce，输出每条回路的 `best_sse`、`best_k`、`best_idx`，避免输出完整评分矩阵。

### 目录结构

```text
prediction/ProcessControl/
├── README.md
└── PIDModelFit/
    ├── README.md
    ├── docs/
    │   ├── design.md
    │   └── test_report.md
    ├── common/
    │   ├── pid_basis_gemm_fit_common.h
    │   ├── pid_basis_gemm_fit_host_impl.h
    │   ├── pid_basis_gemm_fit_kernel.h
    │   ├── pid_basis_gemm_acl_smoke.h
    │   └── pid_basis_gemm_reference.py
    ├── pid_fopdt_basis_gemm_fit/
    ├── pid_ipdt_basis_gemm_fit/
    └── pid_sopdt_basis_gemm_fit/
```

## 如何测试

### 1. 本地 Python 精度测试

```bash
python -m pytest \
  prediction/ProcessControl/PIDModelFit/pid_fopdt_basis_gemm_fit/tests/test_pid_fopdt_basis_gemm_fit.py \
  prediction/ProcessControl/PIDModelFit/pid_ipdt_basis_gemm_fit/tests/test_pid_ipdt_basis_gemm_fit.py \
  prediction/ProcessControl/PIDModelFit/pid_sopdt_basis_gemm_fit/tests/test_pid_sopdt_basis_gemm_fit.py \
  -q
```

结果：

```text
6 passed
```

### 2. Python 语法检查

```bash
python -m compileall -q prediction/ProcessControl/PIDModelFit
```

结果：通过。

### 3. Ascend C 编译与 NPU smoke

已在 node202 / Ascend 910B1 环境验证：

```bash
for op in pid_fopdt_basis_gemm_fit pid_ipdt_basis_gemm_fit pid_sopdt_basis_gemm_fit; do
  cd prediction/ProcessControl/PIDModelFit/$op
  rm -rf build
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
  cmake --build build -j 2
done
```

三个算子均编译通过，并完成 ACL smoke：

```text
PidFopdtBasisGemmFit smoke best_sse=[1, 12] best_k=[1.5, 2] best_idx=[2, 1]
PASSED

PidIpdtBasisGemmFit smoke best_sse=[1, 12] best_k=[1.5, 2] best_idx=[2, 1]
PASSED

PidSopdtBasisGemmFit smoke best_sse=[1, 12] best_k=[1.5, 2] best_idx=[2, 1]
PASSED
```

### 4. 性能验证记录

Ascend 910B1 原型验证中，典型规模下 NPU cached 端到端路径相对 CPU 多线程完整 fit 有明显优势：

| 模型 | 配置 | CPU 多线程完整 fit | NPU cached 端到端 | NPU cold 端到端 |
|------|------|--------------------|-------------------|-----------------|
| FOPDT | `B=64,N=1024,M=256` | `9.22 ms` | `0.43 ms` | `0.87 ms` |
| IPDT | `B=64,N=1024,M=256` | `9.64 ms` | `0.40 ms` | `0.63 ms` |
| SOPDT | `B=64,N=1024,M=256` | `8.46 ms` | `0.57 ms` | `0.87 ms` |
| FOPDT | `B=128,N=1024,M=512` | `31.55 ms` | `0.99 ms` | `1.02 ms` |
| IPDT | `B=128,N=1024,M=512` | `32.63 ms` | `0.64 ms` | `1.30 ms` |
| SOPDT | `B=128,N=1024,M=512` | `33.40 ms` | `0.87 ms` | `1.19 ms` |

详细测试说明见 `prediction/ProcessControl/PIDModelFit/docs/test_report.md`。

## Checklist

- [x] 我的代码遵循这个项目的代码风格
- [x] 我已经自己测试过我的代码
- [x] 我已经更新了相应的文档
- [x] 我已经根据需要更新了对应的变更日志/测试记录
- [x] 我已经在标题中正确使用了类型标签

## 其他信息

本 PR 的核心设计原则是：FOPDT/IPDT/SOPDT 保持工程独立，便于评审、验收、单独测试和后续分别优化；公共底座只复用等价数学逻辑，不影响算子边界。
