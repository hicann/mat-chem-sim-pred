# PID 模型辨识算子组设计文档

## 背景

工业 PID 整定通常先用阶跃响应或运行数据辨识低阶过程模型，再基于模型参数计算 PID 控制器参数。常见模型包括：

- FOPDT：一阶惯性加纯滞后，适合多数自衡对象。
- IPDT：积分加纯滞后，适合液位、库存等积分对象。
- SOPDT：二阶惯性加纯滞后，适合存在两个主导时间常数的对象。

传统 CPU 实现通常逐回路、逐候选参数循环打分。当回路数 `B`、采样点数 `N`、候选数 `M` 同时变大时，核心计算可整理为 `B x N` 与 `N x M` 的矩阵乘，再融合 best-candidate reduce，非常适合 NPU 执行。

## 工程拆分

本交付采用三独立算子结构：

| 算子 | 目录 | 业务含义 |
|------|------|----------|
| `PidFopdtBasisGemmFit` | `pid_fopdt_basis_gemm_fit` | FOPDT 候选模型辨识 |
| `PidIpdtBasisGemmFit` | `pid_ipdt_basis_gemm_fit` | IPDT 候选模型辨识 |
| `PidSopdtBasisGemmFit` | `pid_sopdt_basis_gemm_fit` | SOPDT 候选模型辨识 |

公共目录 `common/` 仅复用数值等价的 reduce 底座：

```text
dot[B, M], basis_norm[M], y_energy[B]
    -> best_sse[B], best_k[B], best_idx[B]
```

三个算子拥有独立 CMake、独立 host/kernel 入口、独立测试和独立文档，便于单独评审、发布和后续优化。

## 数学形式

给定单位增益候选响应 `g_m(t_i)` 和中心化测量输出 `y_b(t_i)`：

```text
dot[b, m] = sum_i y_b(t_i) * g_m(t_i)
basis_norm[m] = sum_i g_m(t_i)^2
y_energy[b] = sum_i y_b(t_i)^2
K_hat[b, m] = dot[b, m] / basis_norm[m]
SSE[b, m] = y_energy[b] - dot[b, m]^2 / basis_norm[m]
best_idx[b] = argmin_m SSE[b, m]
```

其中 `dot` 由 CANN 内置 MatMul 完成，自定义算子完成 `K_hat/SSE/best_idx` 融合 reduce。

## NPU 价值

- 多回路、多候选天然形成 GEMM，大规模时可复用 NPU 矩阵计算能力。
- reduce 算子避免把完整 `SSE[B, M]` 写回 Host，只输出每条回路最优结果。
- 基函数矩阵可按工况缓存，在线场景只需上传新的 `y_centered/y_energy`。
- SOPDT 候选维度更高，NPU 路线的收益最明显。

## 已验证性能

Ascend 910B1 原型验证结果：

| 模型 | 配置 | CPU 多线程完整 fit | NPU cached 端到端 | NPU cold 端到端 |
|------|------|--------------------|-------------------|-----------------|
| FOPDT | `B=64,N=1024,M=256` | `9.22 ms` | `0.43 ms` | `0.87 ms` |
| IPDT | `B=64,N=1024,M=256` | `9.64 ms` | `0.40 ms` | `0.63 ms` |
| SOPDT | `B=64,N=1024,M=256` | `8.46 ms` | `0.57 ms` | `0.87 ms` |
| FOPDT | `B=128,N=1024,M=512` | `31.55 ms` | `0.99 ms` | `1.02 ms` |
| IPDT | `B=128,N=1024,M=512` | `32.63 ms` | `0.64 ms` | `1.30 ms` |
| SOPDT | `B=128,N=1024,M=512` | `33.40 ms` | `0.87 ms` | `1.19 ms` |

精度验证中 `best_sse_mismatch_count=0`，`max_reduce_sse_err` 和 `max_reduce_k_err` 均满足阈值。

## 后续优化方向

- 将 reduce 内层候选扫描改为 UB 分块向量化，降低逐点 GM 读取。
- 支持 `basis_norm` 与候选元数据常驻 Device，进一步降低在线冷启动成本。
- 将 MatMul 与 reduce 纳入图执行，减少 Host 调度开销。
- 为 SOPDT 提供更大规模候选网格 benchmark，突出二阶模型的 NPU 优势。
