# SelectiveScan1D

`SelectiveScan1D` 是面向 SSM/Mamba 类时序预测模型的 Ascend C 自定义算子。它把 PyTorch/torch_npu 路径中由 Python loop 和大量小框架算子表达的 selective scan recurrence 融合为单个 NPU 算子。

## 功能说明

Mamba/SSM block 的核心 scan 递推为：

```text
state_t = exp(delta_t * A) * state_{t-1} + delta_t * B_t * u_t
y_t     = C_t dot state_t + D * u_t
```

时间维 `L` 必须顺序推进，但每个时间步上的 `[D, N]` 状态更新适合在 NPU 上做融合和向量化。该算子按 `(batch, dim)` 分组并行，每个分组内部顺序扫描 `length`。

## 输入输出

| 名称 | 类型 | Shape | 说明 |
|------|------|-------|------|
| `u` | float32 | `[B, L, D]` | 输入序列 |
| `delta` | float32 | `[B, L, D]` | 离散化步长，通常来自 `softplus(dt_proj(...))` |
| `a` | float32 | `[D, N]` | SSM A 参数，稳定模型中通常为负值 |
| `b` | float32 | `[B, L, N]` | selective B |
| `c` | float32 | `[B, L, N]` | selective C |
| `d` | float32 | `[D]` | skip 参数 |
| `output` | float32 | `[B, L, D]` | scan 输出 |

## Host API

```cpp
uint64_t aclnnSelectiveScan1DGetWorkspaceSize(
    int64_t batch, int64_t length, int64_t dim, int64_t state);

int32_t aclnnSelectiveScan1D(
    void* u, void* delta, void* a, void* b, void* c, void* d, void* output,
    int64_t batch, int64_t length, int64_t dim, int64_t state,
    void* workspace, uint64_t workspace_size, void* stream);
```

当前 host API 为轻量 ACL launch 封装，shape 由显式参数传入，workspace 用于保存 device tiling。

## 构建

```bash
cd prediction/ProcessControl/TimeSeriesForecast/selective_scan_1d
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
make -j$(nproc)
```

如本地 CANN 安装路径不是默认路径，可先设置：

```bash
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## 测试

Python reference correctness：

```bash
python -m pytest tests/test_selective_scan_1d.py -q
python -m compileall -q .. selective_scan_1d
```

ACL smoke：

```bash
./build/test_aclnn_selective_scan_1d 0
```

CPU reference benchmark：

```bash
python tests/benchmark_selective_scan_1d.py --batch 1 --length 128 --dim 64 --state 16 --repeat 3
```

NPU benchmark：

```bash
./build/benchmark_selective_scan_1d 0 1 1024 1536 16 10 3
```

## 性能记录

Ascend 910B3 原型验证：

| 场景 | 框架路径 | 自定义算子路径 | speedup | 精度 |
|------|----------|----------------|---------|------|
| scan-only `B=1,D=1536,N=16,L=1024` | torch_npu naive scan `237.7 ms` | vectorized custom scan `27.38 ms` | `8.68x` | 使用硬件 `Exp` 后与 reference 对齐 |
| Mamba block E2E `B=1,L=1024,D_MODEL=768,D_INNER=1536,N=16` | `185.72 ms` | `20.59 ms` | `9.02x` | `max_abs_diff=2.794e-09` |
| MTO forecasting validation loop, 8 windows | `1699.62 ms` | `166.38 ms` | `10.22x` | MSE/MAE diff `0/0` |

详细算法、API、测试口径和当前验证状态见 [docs/algorithm.md](docs/algorithm.md)、[docs/api_reference.md](docs/api_reference.md)、[docs/benchmark.md](docs/benchmark.md) 和 [docs/test_report.md](docs/test_report.md)。

## 迁移边界

本目录只交付 `SelectiveScan1D`。第一批 MR 共包含 10 个独立算子目录；TimesFM 相关算子不属于本批提交。

## TorchAir 补充结论

主测试 `B1,L1024,D1536,N16` 可完整成图，但展开为 18,441 个 FX 节点。TorchAir 为 `27.952 ms`，custom 为 `20.077 ms`，custom 仍快 `1.39x`；首次编译和运行耗时 `883.51 s`。详情见 [docs/benchmark.md](docs/benchmark.md) 和 [第一批总览](../docs/torchair_first_batch_report.md)。
