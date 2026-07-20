# CornnScanFused — fused coRNN (coupled oscillatory RNN) sequence scan (Ascend 910B3, fp32)

Fuses the launch-bound per-timestep coRNN recurrence (Rusch & Mishra, ICLR 2021,
*Coupled Oscillatory Recurrent Neural Network*) into a single NPU kernel launch.

## Op
`CornnScanFused(x:[B,L,IN], weight:[2H+IN,H], bias:[H]) -> output:[B,L,H]`

Per batch `b`, timestep `s` (states `y` position, `z` velocity in R^H), IMEX step:
```
pre = Wy y + Wz z + V x_s + b
z   = z + dt*( tanh(pre) - gamma*y - eps*z )
y   = y + dt*z
out[b,s,:] = y
```
`weight` is the column-packed transposed projection `[2H+IN, H]`:
`[0:H,:]=Wy`, `[H:2H,:]=Wz`, `[2H:2H+IN,:]=V`. `bias:[H]`. Constants `dt=0.042,
gamma=1, eps=1` are baked into the kernel (and matched by the framework baseline).

Two states coupled through tanh each step → no parallel/chunked form; NPU has no
native coRNN primitive. Batches are independent (each AI core owns a contiguous
batch slab, holds `y,z` in UB, no SyncAll / no GM state round-trip / no workspace).

## Three-gate results (Ascend 910B3, fp32, fair framework torch-npu loop baseline)

| Gate | Measurement | Threshold | Result |
|---|---|---|---|
| Correctness (probe, 4 shapes incl. unaligned IN=21/11) | max_diff 2.33e-9 ~ 3.35e-8 | ≤1e-3 | **PASS** |
| Correctness (fused encoder vs fp32 torch) | max_diff 1.49e-7 | ≤1e-3 | **PASS** |
| Component (single layer, B32 L336 H64) | 97.96ms → 1.73ms = **56.73×** | >1.5× | **PASS** |
| E2E (3-layer encoder) | 338.73ms → 6.26ms = **54.15×** | >3× | **PASS** |

**Verdict: STRONG** (all three gates pass).

## Files
- `op_kernel/cornn_scan_fused_kernel.cpp` — fused AscendC kernel
- `op_host/cornn_scan_fused_host.cpp`     — op-host tiling + InferShape + OpDef
- `op_host/cornn_scan_fused_tiling.h`     — tiling data definition
- `msopgen/cornn_scan_fused_msopgen.json` — msopgen IR (3 inputs, 1 output, fp32 ND)
- `tests/cornn_scan_fused_probe.cpp`     — CPU oracle + aclnn correctness/perf probe
- `tests/cornn_e2e.py`                   — torch baseline vs fused encoder (gates 2,3)
- `tests/run_probe.sh`, `tests/run_e2e.sh` — Ascend 910B3 环境 build+run wrappers
- `docs/algorithm.md`                    — design notes
- `docs/api_reference.md`, `docs/benchmark.md`, `docs/test_report.md`
- `docs/cornn_scan_fused_probe_results.json`, `docs/cornn_scan_fused_e2e_results.json`

## Build (Ascend 910B3 环境)
```
python3 cann_ops/scripts/build_msopgen_kernels.py cornn_scan_fused --compute-unit ai_core-ascend910b
```

## TorchAir 补充结论

主测试 `B32,L336,IN11,H64` 可完整成图，产生 3,703 个 FX 节点；TorchAir 为 `3.579 ms`，custom 为 `1.718 ms`，custom 仍快 `2.08x`，首次编译和运行耗时 `92.83 s`。详情见 [docs/benchmark.md](docs/benchmark.md) 和 [第一批总览](../docs/torchair_first_batch_report.md)。
