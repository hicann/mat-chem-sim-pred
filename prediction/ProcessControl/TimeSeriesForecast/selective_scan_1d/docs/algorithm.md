# SelectiveScan1D 算法说明

## 数学定义

给定：

```text
u      in R[B, L, D]
delta  in R[B, L, D]
A      in R[D, N]
B_t    in R[B, L, N]
C_t    in R[B, L, N]
D_skip in R[D]
```

每个 batch/channel 维护一个长度为 `N` 的隐状态：

```text
x[b, d, :]_0 = 0
```

对时间步 `t = 0..L-1`：

```text
decay[b, d, n] = exp(delta[b, t, d] * A[d, n])
update[b, d, n] = delta[b, t, d] * B[b, t, n] * u[b, t, d]
x[b, d, n] = decay[b, d, n] * x[b, d, n] + update[b, d, n]
output[b, t, d] = sum_n x[b, d, n] * C[b, t, n] + u[b, t, d] * D_skip[d]
```

这与 `common/selective_scan_1d_reference.py` 中的 NumPy reference 保持一致。

## 算法流程

1. 将 `(batch, dim)` 展开为 `B * D` 个独立 scan group。
2. 多个 AI Core 按 group 均匀切分任务。
3. 每个 group 先将 `A[d, :]` 拷入 UB，并初始化 `state[N] = 0`。
4. 沿时间维顺序扫描：
   - 从 GM 读取当前 `u[b,t,d]` 和 `delta[b,t,d]`。
   - 将 `B[b,t,:]`、`C[b,t,:]` 拷入 UB。
   - 使用向量指令完成 `Muls -> Exp -> Mul -> Add -> ReduceSum`。
   - 写回 `output[b,t,d]`。
5. 释放本 group 的 local tensor。

## NPU 实现策略

- 保持时间维顺序，以保证 selective scan 递推语义。
- 并行维度选择 `(B, D)`，避免跨时间步并行破坏依赖关系。
- `A[d, :]` 在一个 group 内复用，放入 UB 减少重复 GM 访问。
- `B/C` 按时间步顺序流式搬运，使用 coalesced `DataCopy` 代替逐元素标量读。
- `Exp` 使用硬件向量指令，避免早期近似实现带来的累计误差。
- `ReduceSum` 在 UB 内完成，只写最终 `output`，不物化中间 `state`。

## 边界条件

- `batch/length/dim/state` 必须均为正数，且能放入 `uint32_t` tiling。
- 当前 kernel 按 float32 实现；其他 dtype 需要新增 kernel 分支和精度阈值。
- `state` 会决定 UB buffer 大小。正式扩大支持范围前需要在目标 SOC 上验证 UB 容量和 tiling。
- 输入 `delta` 建议为非负值；这与 Mamba 中 `softplus` 后的离散化步长一致。

## 精度

Python 测试覆盖：

- 小规模确定性样例。
- 随机输入与独立标量 reference 对比。
- skip path、全零输入与 shape 校验。

当前阈值：

```text
max_abs_err <= 1e-5 for small deterministic cases
max_abs_err <= 2e-5 for random float32 cases
```

Ascend 910B3 环境 原型 E2E 记录：

```text
Mamba block max_abs_diff = 2.794e-09
MTO forecasting prediction max_abs_diff = 1.431e-06
MSE/MAE absolute diff = 0 / 0
```
