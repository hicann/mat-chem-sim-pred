# LtcScanFused — fused LTC (Liquid Time-Constant) ODE-solver sequence scan (Ascend 910B3, fp32)

Fuses the launch-bound per-timestep LTC fused semi-implicit ODE solver (Hasani, Lechner,
Amini, Rus & Grosu, AAAI 2021, *Liquid Time-constant Networks*) into a single NPU kernel
launch.

## Op
`LtcScanFused(x:[B,L,IN], weight:[H+IN,H], bias:[5H]) -> output:[B,L,H]`

Per batch `b`, timestep `s` (state `h` in R^H), with `K=6` inner ODE-solver unfolds:
```
wx = W_in x_s                                   (constant across unfolds)
repeat K:
  pre = W_rec h + wx + b
  f   = sigmoid(pre)
  h   = (cm/dt * h + gleak*Eleak + f*A) / (cm/dt + gleak + f)
out[b,s,:] = h
```
`weight` is the column-packed transposed projection `[H+IN, H]`: `[0:H,:]=W_rec`
(hidden-hidden), `[H:H+IN,:]=W_in` (input). `bias` is `[5H]` packed per-neuron:
`[b | gleak | Eleak | cm | A]`. Constants `dt=0.042, K=6` are baked into the kernel
(and matched by the framework baseline).

Each unfold recomputes the hidden-hidden matmul `W_rec h`, so each timestep is K matmuls
plus an elementwise division — a nonlinear gated recurrence with an inner ODE solver →
no parallel/chunked form; NPU has no native LTC primitive. Batches are independent (each
AI core owns a contiguous batch slab, holds `h` in UB, no SyncAll / no GM state
round-trip / no workspace).

## Three-gate results (Ascend 910B3, fp32, fair framework torch-npu loop baseline)

| Gate | Measurement | Threshold | Result |
|---|---|---|---|
| Correctness (probe, 4 shapes incl. unaligned IN=21/11) | max_diff 1.97e-6 ~ 3.92e-6 | ≤1e-3 | **PASS** |
| Correctness (fused encoder vs fp32 torch) | max_diff 2.53e-7 | ≤1e-3 | **PASS** |
| Component (single layer, B32 L336 H64) | 379.21ms → 4.60ms = **82.44×** | >1.5× | **PASS** |
| E2E (3-layer encoder) | 1134.71ms → 14.76ms = **76.90×** | >3× | **PASS** |

**Verdict: STRONG** (all three gates pass). The K inner unfolds multiply the per-step
launch overhead in the framework baseline (gap-hunt 13992×/4742×, largest of the family),
so fusing all K·L steps into one kernel yields a very large speedup.

## Files
- `op_kernel/ltc_scan_fused_kernel.cpp` — fused AscendC kernel
- `op_host/ltc_scan_fused_host.cpp`     — op-host tiling + InferShape + OpDef
- `op_host/ltc_scan_fused_tiling.h`     — tiling data definition
- `msopgen/ltc_scan_fused_msopgen.json` — msopgen IR (3 inputs, 1 output, fp32 ND)
- `tests/ltc_scan_fused_probe.cpp`     — CPU oracle + aclnn correctness/perf probe
- `tests/ltc_e2e.py`                   — torch baseline vs fused encoder (gates 2,3)
- `tests/run_probe.sh`, `tests/run_e2e.sh` — Ascend 910B3 环境 build+run wrappers
- `docs/algorithm.md`                       — design notes
- `docs/api_reference.md`, `docs/benchmark.md`, `docs/test_report.md`
- `docs/ltc_scan_fused_probe_results.json`, `docs/ltc_scan_fused_e2e_results.json`

## Build (Ascend 910B3 环境)
```
python3 cann_ops/scripts/build_msopgen_kernels.py ltc_scan_fused --compute-unit ai_core-ascend910b
```

## TorchAir 补充结论

主测试 `B32,L336,IN11,H64,K6` 可完整成图，但展开为 24,878 个 FX 节点。TorchAir 为 `10.227 ms`，custom 为 `4.909 ms`，custom 仍快 `2.08x`；首次编译和运行耗时 `616.61 s`。详情见 [docs/benchmark.md](docs/benchmark.md) 和 [第一批总览](../docs/torchair_first_batch_report.md)。
