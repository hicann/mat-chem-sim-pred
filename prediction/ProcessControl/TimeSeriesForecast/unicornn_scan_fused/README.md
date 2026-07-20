# UnicornnScanFused — fused UnICORNN (undamped independent controlled oscillatory RNN) sequence scan (Ascend 910B3, fp32)

Fuses the launch-bound per-timestep UnICORNN recurrence (Rusch & Mishra, ICML 2021,
*UnICORNN: A recurrent model for learning very long time dependencies*) into a single
NPU kernel launch.

## Op
`UnicornnScanFused(x:[B,L,IN], weight:[IN+2,H], bias:[H]) -> output:[B,L,H]`

Per batch `b`, timestep `s` (states `y` position, `z` velocity in R^H):
```
arg   = w (.) y + V x_s + b        (w is DIAGONAL recurrent: elementwise)
sig_c = sigmoid(c)                 (per-neuron control / time-constant step size)
z     = z - dt*sig_c*( tanh(arg) + alpha*y )
y     = y + dt*sig_c*z
out[b,s,:] = y
```
`weight` is the column-packed transposed projection `[IN+2, H]`:
`[0:IN,:]=V` (input projection), `[IN,:]=w` (diagonal recurrent weight),
`[IN+1,:]=c` (per-neuron control). `bias:[H]=b`. Constants `dt=0.042, alpha=1`
are baked into the kernel (and matched by the framework baseline).

Hidden-hidden coupling is **diagonal** (independent per channel) — only the input
projection `V x_s` is a matmul. Nonlinear tanh feedback → no parallel/chunked form;
NPU has no native UnICORNN primitive. Batches are independent (each AI core owns a
contiguous batch slab, holds `y,z` in UB, no SyncAll / no GM state round-trip / no
workspace).

## Three-gate results (Ascend 910B3, fp32, fair framework torch-npu loop baseline)

| Gate | Measurement | Threshold | Result |
|---|---|---|---|
| Correctness (probe, 4 shapes incl. unaligned IN=21/11) | max_diff 5.53e-10 ~ 2.98e-8 | ≤1e-3 | **PASS** |
| Correctness (fused encoder vs fp32 torch) | max_diff 2.68e-7 | ≤1e-3 | **PASS** |
| Component (single layer, B32 L336 H64) | 79.76ms → 0.496ms = **160.68×** | >1.5× | **PASS** |
| E2E (3-layer encoder) | 243.79ms → 2.43ms = **100.43×** | >3× | **PASS** |

**Verdict: STRONG** (all three gates pass). Highest speedup in the family — the
diagonal recurrence has almost no matmul, so the per-timestep torch loop is
extremely launch-bound and fusion into one kernel yields the largest gain.

## Files
- `op_kernel/unicornn_scan_fused_kernel.cpp` — fused AscendC kernel
- `op_host/unicornn_scan_fused_host.cpp`     — op-host tiling + InferShape + OpDef
- `op_host/unicornn_scan_fused_tiling.h`     — tiling data definition
- `msopgen/unicornn_scan_fused_msopgen.json` — msopgen IR (3 inputs, 1 output, fp32 ND)
- `tests/unicornn_scan_fused_probe.cpp`     — CPU oracle + aclnn correctness/perf probe
- `tests/unicornn_e2e.py`                   — torch baseline vs fused encoder (gates 2,3)
- `tests/run_probe.sh`, `tests/run_e2e.sh` — Ascend 910B3 环境 build+run wrappers
- `docs/algorithm.md`                       — design notes
- `docs/api_reference.md`, `docs/benchmark.md`, `docs/test_report.md`
- `docs/unicornn_scan_fused_probe_results.json`, `docs/unicornn_scan_fused_e2e_results.json`

## Build (Ascend 910B3 环境)
```
python3 cann_ops/scripts/build_msopgen_kernels.py unicornn_scan_fused --compute-unit ai_core-ascend910b
```

## TorchAir 补充结论

主测试 `B32,L336,IN11,H64` 可完整成图，产生 4,715 个 FX 节点；TorchAir 为 `3.544 ms`，custom 为 `0.521 ms`，custom 仍快 `6.80x`，首次编译和运行耗时 `250.93 s`。详情见 [docs/benchmark.md](docs/benchmark.md) 和 [第一批总览](../docs/torchair_first_batch_report.md)。
