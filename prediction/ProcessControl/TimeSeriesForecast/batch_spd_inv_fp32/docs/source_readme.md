# BatchSpdInvFp32 — on-device batched small SPD inverse for Koopa DMD

Status: **experimental** · Value class: **unsupported -> supported**
Device: Ascend 910B3, CANN 8.1.RC1.alpha001

This package delivers a custom CANN operator that removes Koopa's DMD-step CPU
fallback. Koopa's `KPLayer` fits its Koopman matrix with `torch.linalg.lstsq`,
which is unsupported on the NPU and silently runs on CPU (1.3–4.3 s per call;
~5–15 s per `forecast()` across 3 blocks). The only piece forcing the fallback is
a tiny `m×m` SPD (Gram) inverse — this operator provides exactly that on-device.

```
G  = bmm(x, x^T)            # native NPU bmm, ~0.04ms
Gi = BatchSpdInvFp32(G)     # this operator
K  = bmm(x^T, bmm(Gi, y))   # native NPU bmm, ~0.04ms
```

## Layout

```
batch_spd_inv_fp32_experimental/
├── README.md                          this file
├── FILE_INDEX.json                    manifest (every file + size + role)
├── batch_spd_inv_fp32_msopgen.json    operator IR (1 in g, 1 out gi; fp32 ND)
├── source/
│   ├── batch_spd_inv_fp32_ascendc.cpp kernel (pure-Vector LDL^T + tri-inverse)
│   └── build_batch_spd_inv.py         msopgen build driver (tiling + infershape)
├── tests/
│   ├── koopa_validate_ctypes.py       correctness + isolated DMD-vs-lstsq bench
│   └── koopa_e2e.py                   full Koopa forecast() E2E, stock vs custom
├── docs/
│   ├── DESIGN.md                      full design / algorithm / bug fixes / results
│   ├── summary.md                     one-page summary
│   ├── batch_spd_inv_fp32_results.json    raw correctness + isolated latency
│   └── batch_spd_inv_fp32_e2e_koopa.json  raw Koopa E2E results
└── generated/                         built artifacts (aclnn wrappers, .so, op_host)
    ├── autogen/                        aclnn_*.cpp/.h, op_proto.h
    ├── op_api_include/                 public aclnn API header
    ├── op_host_src/                    host tiling + shape-inference source
    ├── op_host/                        op_host install .so
    └── libs/                           libcust_opapi.so, libascend_all_ops.so
```

## Interface

```
aclnnBatchSpdInvFp32(g, gi)
  g  : float32, [B, m, m], ND   batched SPD matrices (Gram = x x^T)
  gi : float32, [B, m, m], ND   per-batch inverse
```
`m` small (Koopa 3–7; cap 64). Batches spread across AIV cores.

## Build (Ascend 910B3 环境)

```bash
# CANN env sourced; from the op-dev tree
python3 source/build_batch_spd_inv.py     # msopgen -> inject tiling/infershape -> build.sh
# produces generated/libs/libcust_opapi.so (custom op registered, aclnnBatchSpdInvFp32 callable)
```

## Run / validate (Ascend 910B3 环境, NPU visible)

```bash
python3 tests/koopa_validate_ctypes.py    # correctness vs torch.linalg.inv/lstsq + isolated bench
python3 tests/koopa_e2e.py                # Koopa forecast() stock(lstsq CPU) vs custom on-device DMD
```

The ctypes harness loads `generated/libs/libcust_opapi.so` and calls the two-phase
aclnn entry (`aclnnBatchSpdInvFp32GetWorkspaceSize` + `aclnnBatchSpdInvFp32`).

## Results at a glance (Ascend 910B3 环境 / 910B3)

| Check | Result | Gate |
|---|---|---|
| `G^-1` vs `torch.linalg.inv` | < 1e-7 | 1e-5 |
| DMD `K` vs `torch.linalg.lstsq` | < 7e-8 | 1e-5 |
| reconstruction residual `\|x·K − y\|` | < 1e-6 | — |
| isolated DMD, `B=32,m=7,E=128` | **4341.75 ms CPU fallback -> 0.339 ms custom NPU, 12823.56x**, diff `6.80e-08` | > 100x |
| Koopa forecast E2E, `seq=336,pred=96,B=32` | **5300.64 ms CPU fallback -> 8.80 ms custom NPU, 602.19x**, diff `2.38e-07` | > 50x |

## Notes

- **Algorithm:** LDL^T factorization (no sqrt) + triangular inverse, scalar-only on
  the Vector unit. No Cube / Matmul API, no cross-core sync. See `docs/DESIGN.md` §3.
- **Multi-core cache-coherency fix:** scalar `GetValue`/`SetValue` directly on GM is
  not L1-coherent across AIV cores (B>1 produced garbage). Each batch is DMA'd into
  UB, worked on locally, and DMA'd back. See `docs/DESIGN.md` §4.
- **We do not fuse the surrounding matmuls** — they are already trivial native bmm;
  the operator is the single missing primitive (the `m×m` SPD inverse).
- **Out of scope (model-side):** Koopa's `FourierFilter` uses a complex64
  `index_put`, also unsupported on NPU. The E2E harness patches it identically on
  both stock and custom paths so the reported speedup isolates the DMD inverse.
- The stable 3-op package (`route_b_custom_ops`, validated by
  `validate_route_b_package.py`) is untouched by this experimental deliverable.


## Robustness (2026-06-23)

The kernel now clamps each LDL^T pivot to `max(d[j], 1e-6*max|diag g|)` so a
rank-deficient / near-singular Gram (collinear or constant channels, short
context) yields a finite, bounded result instead of Inf/NaN that would poison the
whole forecast. CPU oracle `tests/batch_spd_inv_cpu_ref.cpp` (no NPU needed)
verifies: well-conditioned accuracy is unchanged (bit-identical to the unguarded
kernel, max_diff vs fp64 oracle <= 8.8e-7), and rank-deficient inputs stay finite.
On-device check: `python3 tests/koopa_validate_ctypes.py --mode robust`.
Rebuild on Ascend 910B3 环境 with `build_batch_spd_inv_node.sh` (the `generated/` artifacts
predate this change).
