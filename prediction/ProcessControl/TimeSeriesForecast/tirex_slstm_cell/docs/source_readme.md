# TirexSlstmCell — vectorized fused sLSTM scan for TiRex

Status: **experimental** · Value class: **supported -> faster**
Device: Ascend 910B3, CANN 8.1.RC1.

This package delivers a custom CANN operator that replaces the recurrent sLSTM
scan of **NX-AI/TiRex** (xLSTM, 2025 — a hot non-transformer time-series
foundation model; its block0 is 100% sLSTM). The stock torch_npu scan runs one
launch per timestep and is dominated by the pointwise gate math; this operator
fuses the whole sequence into a single launch with vectorized gates spread
across all AI vector cores.

```
output, final = TirexSlstmCell(input, recurrent_kernel, bias, initial_state)
# input[B,S,4H] (gate input projections, order [i,f,z,o])
# recurrent_kernel[heads,Hd,4Hd], bias[4H], initial_state[4,B,H] ([h,c,n,m])
# output[B,S,H] (h sequence), final_state[4,B,H]
```

## Results at a glance (Ascend 910B3 环境, B=64 S=64 H=512 heads=4)

| metric | value |
|---|---|
| correctness vs CPU oracle (probe) | max_diff 4.77e-7 |
| correctness vs fp32 torch ref, **real TiRex weights+data** | max_diff 1.45e-6 |
| custom op (scan only) | **3.26–3.66 ms** |
| torch_npu cell scan (`cell_impl`) | 52.3 ms → **14.3×** |
| torch_npu full sLSTM layer (gate-proj + scan + group_norm) | 53.9 ms → **10.5×** end-to-end (custom layer 5.15 ms) |
| old scalar custom kernel (baseline) | 3738 ms → **940×** |

Tuning: switching the host `block_dim` from the original `min(batch,32)` to
`min(batch*num_heads, GetCoreNumAiv())` (use all vector cores) took B=64 S=64
from 4.20 ms to 3.26 ms and helped small-batch shapes most (B=16: 1.96→1.04 ms;
B=8/H256: 0.58→0.21 ms).

## Layout

```
tirex_slstm_cell_vectorized_experimental/
├── README.md                          this file
├── FILE_INDEX.json                    manifest (every file + size + role)
├── tirex_slstm_cell_msopgen.json      operator IR (4 in, 2 out; fp32 ND)
├── source/
│   ├── tirex_slstm_cell_ascendc.cpp   kernel (vectorized fused scan, kRowBlk=8)
│   └── tirex_slstm_cell_ascendc.scalar_bak.cpp  original scalar kernel (3738 ms baseline)
├── op_host/
│   └── tirex_slstm_cell_host.cpp       host tiling (aiv-based block_dim)
├── tests/
│   ├── tirex_slstm_cell_probe.cpp     C++ harness: aclnn op vs CPU oracle Ref(), bench
│   ├── tirex_slstm_e2e.py             ctypes-aclnn E2E: real TiRex block0, op vs torch
│   ├── run_probe.sh                   build vendor env + run probe (correctness+bench)
│   ├── bench_tirex.sh                 multi-shape bench sweep
│   └── run_e2e.sh                     run the real-model E2E harness
└── docs/
    ├── DESIGN.md                      math, interface, parallelization, sync, why-it-wins
    ├── tirex_slstm_probe_results.json probe correctness+latency across shapes
    └── tirex_slstm_e2e_results.json   real-TiRex E2E (cell 14.3×, layer 10.5×)
```

## Build / run (Ascend 910B3 环境)

The kernel builds via msopgen into
`build/msopgen_tirex_slstm_cell/build_out/`. The probe and E2E harness need the
custom-opp vendor path + `LD_LIBRARY_PATH` set; `tests/run_probe.sh` and
`tests/run_e2e.sh` set these up (vendor symlinks to `op_impl/.../kernel` and
`libcust_opmaster_rt2.0.so`, then `ASCEND_CUSTOM_OPP_PATH`). The formal host
implementation is `op_host/tirex_slstm_cell_host.cpp`; it contains the
msopgen scaffold plus the aiv-based `block_dim` adaptation used by this operator.

probe usage: `./tirex_slstm_cell_probe B S H Heads repeats check`
e.g. `./tirex_slstm_cell_probe 64 64 512 4 8 1`.
