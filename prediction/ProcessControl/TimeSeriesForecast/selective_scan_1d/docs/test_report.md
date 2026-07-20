# SelectiveScan1D 测试报告

## 本地验证

环境：

```text
Windows workspace: C:\tslib\mat-chem-sim-pred
Python: repository default python
CANN toolkit: not installed locally
```

已执行：

```bash
python -m compileall -q prediction\ProcessControl\TimeSeriesForecast
python -m pytest prediction\ProcessControl\TimeSeriesForecast\selective_scan_1d\tests\test_selective_scan_1d.py -q
python prediction\ProcessControl\TimeSeriesForecast\selective_scan_1d\tests\benchmark_selective_scan_1d.py --batch 1 --length 16 --dim 8 --state 4 --repeat 2 --warmup 1
git diff --check
```

结果：

| 项目 | 结果 |
|------|------|
| Python compileall | PASS |
| Python correctness tests | PASS, `10 passed` |
| CPU reference benchmark | PASS, 输出 `nan_count=0` |
| whitespace check | PASS，仅有 Windows CRLF 提示 |

CPU reference benchmark sample：

```text
name,shape,cpu_ref_mean_ms,cpu_ref_min_ms,cpu_ref_max_ms,nan_count,output_sum
SelectiveScan1D,1x16x8xN4,0.449200,0.443100,0.455300,0,0.094520
```

## Ascend 910B3 复验

迁移目录已完成 msopgen 生成、C++ 构建、ACLNN runtime、正确性和正式 shape benchmark。复现命令：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd prediction/ProcessControl/TimeSeriesForecast/selective_scan_1d
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
./build/test_aclnn_selective_scan_1d 0
./build/benchmark_selective_scan_1d 0 1 1024 1536 16 10 3
```

## 既有研发验证记录

来自 `C:\tslib\cann_ops` 的 Ascend 910B3 原型结果：

| 场景 | speedup | 精度 |
|------|---------|------|
| scan-only custom vectorized vs torch_npu naive scan | `8.68x` | hardware `Exp` 后对齐 reference |
| Mamba block E2E | `9.02x` | `max_abs_diff=2.794e-09` |
| MTO forecasting validation loop | `10.22x` | MSE/MAE diff `0/0` |

迁移目录的 CANN 构建和 runtime 验证已完成，结果已回填本文件和 `docs/benchmark.md`。

## TorchAir fullgraph 验证

| 项目 | 结果 |
|---|---|
| 主测试 shape | `B1,L1024,D1536,N16` |
| 完整成图 | PASS，18,441 FX 节点 |
| 首次编译+运行 | `883.51 s` |
| eager / TorchAir / custom | `343.825 / 27.952 / 20.077 ms` |
| custom vs TorchAir | `1.39x` |
| graph/custom max diff | `0 / 4.66e-09` |

详情见 [benchmark.md](benchmark.md) 和 [第一批总览](../../docs/torchair_first_batch_report.md)。
