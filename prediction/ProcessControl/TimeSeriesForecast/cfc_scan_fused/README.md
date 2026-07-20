# CfcScanFused — fused CfC (Closed-form Continuous-time) sequence scan (Ascend 910B3, fp32)

Fuses the launch-bound per-timestep CfC recurrence (Hasani et al., 2022, *Closed-form
Continuous-time Neural Networks*) into a single NPU kernel launch.

## Op
`CfcScanFused(x:[B,L,IN], weight:[H+IN,3H], bias:[3H]) -> output:[B,L,H]`

Per batch `b`, timestep `s` (state `h` in R^H):
```
z    = concat(h, x_s)                  # [H+IN]
ff1  = tanh   (Wf1 z + bf1)            # candidate A
ff2  = tanh   (Wf2 z + bf2)            # candidate B
gate = sigmoid(Wt  z + bt )            # time-interpolation gate
h    = ff1*(1-gate) + ff2*gate
out[b,s,:] = h
```
`weight` is the column-packed transposed projection `[H+IN, 3H]`:
`[:,0:H]=Wf1`, `[:,H:2H]=Wf2`, `[:,2H:3H]=Wt`. `bias = concat(bf1,bf2,bt)`.

`h` feeds back through tanh/sigmoid each step → no parallel/chunked form; NPU has no
native CfC primitive. Batches are independent (each AI core owns a contiguous batch
slab, holds `h` in UB, no SyncAll / no GM state round-trip / no workspace).

## Three-gate results (Ascend 910B3, fp32, fair framework torch-npu loop baseline)

| Gate | Measurement | Threshold | Result |
|---|---|---|---|
| Correctness (probe, 4 shapes incl. unaligned IN=21/11) | max_diff 9.87e-8 ~ 2.38e-7 | ≤1e-3 | **PASS** |
| Correctness (fused encoder vs fp32 torch) | max_diff 2.33e-7 | ≤1e-3 | **PASS** |
| Component (single layer, B32 L336 H64) | 104.39ms → 1.23ms = **84.83×** | >1.5× | **PASS** |
| E2E (3-layer encoder) | 330.0ms → 4.70ms = **70.26×** | >3× | **PASS** |

**Verdict: STRONG** (all three gates pass).

## Files
- `op_kernel/cfc_scan_fused_kernel.cpp` — fused AscendC kernel
- `op_host/cfc_scan_fused_host.cpp`     — op-host tiling + InferShape + OpDef
- `op_host/cfc_scan_fused_tiling.h`     — tiling data definition
- `msopgen/cfc_scan_fused_msopgen.json` — msopgen IR (3 inputs, 1 output, fp32 ND)
- `tests/cfc_scan_fused_probe.cpp`     — CPU oracle + aclnn correctness/perf probe
- `tests/cfc_e2e.py`                   — torch baseline vs fused encoder (gates 2,3)
- `tests/run_probe.sh`, `tests/run_e2e.sh` — Ascend 910B3 环境 build+run wrappers
- `docs/algorithm.md`                  — design notes
- `docs/api_reference.md`, `docs/benchmark.md`, `docs/test_report.md`
- `docs/cfc_scan_fused_probe_results.json`, `docs/cfc_scan_fused_e2e_results.json`

## Build (Ascend 910B3 环境)
```
python3 cann_ops/scripts/build_msopgen_kernels.py cfc_scan_fused --compute-unit ai_core-ascend910b
```

## TorchAir 补充结论

主测试 `B32,L336,IN11,H64` 可完整成图，产生 7,062 个 FX 节点；TorchAir 为 `8.090 ms`，custom 为 `1.265 ms`，custom 仍快 `6.40x`，首次编译和运行耗时 `154.54 s`。详情见 [docs/benchmark.md](docs/benchmark.md) 和 [第一批总览](../docs/torchair_first_batch_report.md)。
