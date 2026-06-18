# PID 模型辨识算子组

本目录提供三个工程上独立的 Ascend C 算子，分别面向工业 PID 整定中常见的 FOPDT、IPDT、SOPDT 过程模型辨识。三者共享 basis-GEMM 评分底座，但拥有独立目录、独立 CMake、独立 host/kernel 入口、独立文档和独立测试。

| 算子目录 | 公开接口 | 模型语义 |
|----------|----------|----------|
| [pid_fopdt_basis_gemm_fit](pid_fopdt_basis_gemm_fit/README.md) | `aclnnPidFopdtBasisGemmFit` | 一阶惯性加纯滞后模型 |
| [pid_ipdt_basis_gemm_fit](pid_ipdt_basis_gemm_fit/README.md) | `aclnnPidIpdtBasisGemmFit` | 积分加纯滞后模型 |
| [pid_sopdt_basis_gemm_fit](pid_sopdt_basis_gemm_fit/README.md) | `aclnnPidSopdtBasisGemmFit` | 二阶惯性加纯滞后模型 |

## 总体算法

每个模型族先把候选参数网格转换为单位增益响应基函数矩阵 `basis_t[N, M]`，再对 `B` 条回路的中心化输出 `y_centered[B, N]` 执行：

```text
dot[B, M] = y_centered[B, N] x basis_t[N, M]
K_hat[b, m] = dot[b, m] / basis_norm[m]
SSE[b, m] = y_energy[b] - dot[b, m]^2 / basis_norm[m]
best[b] = argmin_m SSE[b, m]
```

其中 GEMM 使用 CANN 内置 MatMul，三个自定义算子负责模型族语义下的 best-candidate reduce，输出 `best_sse[B]`、`best_k[B]` 和 `best_idx[B]`。

## 为什么拆成三个算子

- FOPDT/IPDT/SOPDT 是 PID 整定中的三类不同过程模型，业务含义、参数网格和结果解释不同。
- PR 评审、交付验收和后续维护时，可以单独构建、测试、压测和替换任意一个模型族。
- 公共底座只复用数学等价的 `dot -> K/SSE/best` reduce，避免重复维护底层数值逻辑。

## 已验证结果

在 Ascend 910B1 环境上，三类模型均已完成 CPU 单线程、CPU 多线程、NPU cached/cold basis 路径对比。中等规模 `B=64, N=1024, M=256` 下，NPU cached 端到端耗时约 `0.40-0.57 ms`，CPU 多线程完整 fit 约 `8.46-9.64 ms`，精度误差通过阈值校验。

详细设计与测试材料：

- [设计文档](docs/design.md)
- [测试报告](docs/test_report.md)
