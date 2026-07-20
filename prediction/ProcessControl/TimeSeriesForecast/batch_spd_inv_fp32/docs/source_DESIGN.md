# BatchSpdInvFp32 — batched small SPD matrix inverse (fp32) for Koopa DMD

Status: experimental, validated on Ascend 910B3, CANN 8.1.RC1.alpha001.
Value class: **unsupported -> supported** (NPU has no on-device path; the stock
model silently falls back to CPU).

## 1. Motivation — the Koopa DMD bottleneck

Koopa (NeurIPS'23, Koopman-operator time-series model, non-Transformer) embeds the
series into a linear state space and, in every `KPLayer`, fits a Koopman transition
matrix `K` by solving the underdetermined linear system

```
x @ K = y          x, y : [B, m, E]   (m snapshots, E = dynamic_dim, m < E)
```

via `K = torch.linalg.lstsq(x, y).solution`.

`torch.linalg.lstsq` (and `torch.linalg.solve` / `torch.linalg.inv`) are **not
implemented on the NPU backend** and silently fall back to CPU:

```
Warning: CAUTION: The operator 'aten::linalg_lstsq.out' is not currently supported
on the NPU backend and will fall back to run on the CPU.
```

Measured cost of a single fallback call: **1260–4341 ms** (B=16–32). Koopa runs
3 `KPLayer` blocks per forward, so a single `forecast()` spends **~5–15 s** almost
entirely inside lstsq.

### Where the time actually goes

lstsq min-norm closed form for the underdetermined case is

```
K = x^T (x x^T)^-1 y
```

Profiling shows:
- the two big matmuls `x @ x^T` ([B,m,E]·[B,E,m]) and `x^T @ (G^-1 @ y)` run as
  native `torch.bmm` on the NPU in **~0.04 ms** — fully supported, fast;
- the only thing that forces the CPU fallback is the tiny **m×m Gram inverse**
  `(x x^T)^-1` (m ∈ [3,7] for Koopa).

So the missing primitive is exactly **a batched small-SPD-matrix inverse**. Provide
that on-device and the entire DMD solve stays on the NPU:

```
G  = bmm(x, x^T)            # native, ~0.04ms
Gi = BatchSpdInvFp32(G)     # custom op, this deliverable
K  = bmm(x^T, bmm(Gi, y))   # native, ~0.04ms
```

We deliberately do **not** fuse the matmuls into the kernel — they are already
trivial on the NPU; fusing them only adds risk for ~zero gain. The operator is the
single missing primitive and nothing more.

## 2. Interface

```
aclnnBatchSpdInvFp32(g, gi)
  g  : float32, [B, m, m], ND   batched SPD matrices (Gram = x x^T)
  gi : float32, [B, m, m], ND   per-batch inverse
```

`m` is small (Koopa 3–7; supported cap 64). `B` is the batch.

## 3. Algorithm — LDL^T factorization + triangular inverse (no sqrt, no Cube)

Per batch element, with `g` SPD and `m×m`:

```
g = L D L^T            L unit lower-triangular, D diagonal (positive)
g^-1 = L^-T D^-1 L^-1
```

Steps (all scalar arithmetic — only +, -, *, / ; no square roots):
1. **LDL^T factorization**
   `D[j] = g[j][j] - sum_{k<j} L[j][k]^2 D[k]`
   `L[i][j] = (g[i][j] - sum_{k<j} L[i][k] D[k] L[j][k]) / D[j]`,  i > j
2. **D^-1** — scalar reciprocals `1/D[k]`.
3. **L^-1 = Minv** by forward substitution (unit lower-triangular, no division).
4. **g^-1 = Minv^T D^-1 Minv**, assembled symmetrically:
   `gi[i][j] = sum_{k>=max(i,j)} Minv[k][i] D^-1[k] Minv[k][j]`.

Why LDL^T and not Cholesky: LDL^T avoids `sqrt` entirely (Cholesky needs a square
root per pivot), keeping the kernel to plain scalar +,-,*,/ and avoiding the small
numerical noise / extra op of a scalar sqrt. For SPD inputs it is equally stable.

Why pure Vector (no Cube / Matmul API): m is tiny (3–7). A systolic-array matmul on a
3×3–7×7 matrix is pure overhead. Scalar triangular work on the Vector unit is the
right tool, has no cross-core synchronization, and is the simplest/most robust path.

## 4. Parallelization & the cross-core cache-coherency fix

Batches are distributed across AIV cores by `blockIdx`/`blockNum` (each core handles
`b = blk, blk+nblk, blk+2*nblk, ...`). `blockDim` is set to `min(batch, nAIV)`.

### Critical bug: scalar GM access is NOT cache-coherent across cores

The first version did the scalar `GetValue`/`SetValue` of the LDL^T arithmetic
**directly on global memory** (`g`/`gi` in HBM). With B>1 (multiple AIV cores
active) this produced garbage: per-batch residual `|g·gi − I|` blew up to
`~1.0–2.8e3` on ~half the batches, while single-batch was correct.

Root cause: the Vector core's scalar unit accesses to **global memory are not L1
cache-coherent across AIV cores** — each core can read stale/garbage values written
by another core. Scalar GM read/write in a multi-core kernel is unsafe.

Fix (final kernel):
1. `DataCopyPad` the per-batch `g[b]` from GM into a **local UB buffer** at the top
   of the loop;
2. do *all* scalar work (LDL^T, L^-1, D^-1, gi assembly) on **UB tensors** — each
   core has its own private UB copy;
3. `DataCopyPad` the result `gi[b]` from UB back to GM;
4. bracket the DMA phases with `pipe_barrier(PIPE_ALL)`.

After the fix every batch passes (worst-case residual 1.26e-7). General lesson for
multi-core AscendC: **never do scalar accesses directly to GM in a multi-core loop;
always stage through UB.**

## 5. Workspace / UB layout

Per-core UB buffers (allocated once in `Init`):
- `gBuf_`  : `roundup(m*m, 8)` floats — DMA staging for `g[b]`
- `giBuf_` : `roundup(m*m, 8)` floats — DMA staging for `gi[b]`
- `lBuf_`  : `m*m` floats — L (unit lower-triangular)
- `invBuf_`: `m*m` floats — Minv = L^-1
- `dBuf_`  : `m` floats — diagonal D
- `diBuf_` : `m` floats — D^-1

No system/user GM workspace is needed beyond the in/out tensors.

## 6. Build & run

- IR: `batch_spd_inv_fp32_msopgen.json` (1 input `g`, 1 output `gi`, both fp32 ND).
- Build driver: `build_batch_spd_inv.py` — runs msopgen, injects `TilingFunc`
  (extracts `batch`, `m` from the input shape; sets `blockDim = min(batch, nAIV)`;
  only a `{batch, m}` tiling struct — no TCubeTiling) and `InferShape`
  (output shape == input shape), then `build.sh` -> `libcust_opapi.so`.
- Harness: `koopa_validate_ctypes.py` (ctypes -> `aclnnBatchSpdInvFp32`):
  - `spd_inv(g)` — call the op on device;
  - `dmd_custom(x,y)` — `bmm(x,x^T) -> spd_inv -> bmm(x^T, bmm(Gi,y))`;
  - `correctness()` — vs `torch.linalg.inv` and `torch.linalg.lstsq`;
  - `bench()` — isolated DMD path vs lstsq CPU fallback.
- E2E: `koopa_e2e.py` — full Koopa `forecast()`, stock lstsq vs custom DMD path.

## 7. Results (Ascend 910B3 环境 / 910B3)

### Correctness (`docs/batch_spd_inv_fp32_results.json`)
Shapes B={8,16,32}, m={3,5,6,7}, E={64,128,256}:
- `max_diff(custom G^-1, torch.linalg.inv)` < **1e-7**
- `max_diff(custom K, torch.linalg.lstsq)` < **7e-8**
- reconstruction residual `|x·K − y|` < **1e-6**
(gate 1e-5 — passed by ~2 orders of magnitude)

### Isolated DMD path vs lstsq CPU fallback
| B  | m | E   | custom (ms) | lstsq CPU (ms) | speedup  |
|----|---|-----|-------------|----------------|----------|
| 16 | 3 | 128 | 0.386       | 932.5          | 2419x    |
| 16 | 6 | 128 | 0.328       | 1799.4         | 5481x    |
| 32 | 7 | 128 | 0.339       | 4341.7         | 12824x   |
| 32 | 5 | 256 | 0.337       | 3282.0         | 9728x    |
(gate >100x — passed)

### Koopa forecast E2E (`docs/batch_spd_inv_fp32_e2e_koopa.json`)
3 KPLayer blocks, stock lstsq (CPU fallback) vs custom on-device DMD:
| seq_len | pred_len | B  | m | stock (ms) | custom (ms) | speedup | max_diff |
|---------|----------|----|---|------------|-------------|---------|----------|
| 336     | 96       | 32 | 3 | 5300.6     | 8.80        | 602x    | 2.4e-7   |
| 720     | 96       | 32 | 7 | 14730.0    | 9.77        | 1508x   | 3.0e-7   |
| 512     | 96       | 16 | 5 | 5300.0     | 8.89        | 596x    | 2.4e-7   |
(gate >50x — passed)

Note: the E2E harness also patches Koopa's `FourierFilter` to mask via a
real-valued mask instead of a complex64 `index_put` (the latter is a *separate*,
model-side NPU gap unrelated to this operator). The patch is applied identically to
both the stock and custom paths, so the reported speedup isolates the DMD inverse.

## 8. Honest positioning

- This converts Koopa's DMD from **CPU-fallback-only (5–15 s)** to **fully
  on-device (~9 ms)** at fp32 accuracy (diff ~2.4e-7 vs CPU lstsq) — a clean
  "unsupported -> supported" win, ~600–1500x E2E.
- The operator is the single missing primitive (the m×m SPD inverse); the
  surrounding matmuls are native NPU bmm.
- Remaining model-side gap (out of scope here): Koopa's `FourierFilter` uses a
  complex64 `index_put`, also unsupported on NPU — a separate model-side fix.

## 9. Robustness hardening (2026-06-23): LDL^T pivot floor

Toward promoting this op from experimental to stable, the kernel was hardened
against rank-deficient / near-singular Gram inputs.

### The gap
`g = x x^T` is generically SPD and full-rank for Koopa (m << E), but a real
forecast window can be degenerate: collinear or constant channels, or a context
too short to give m independent snapshots. Then an LDL pivot `d[j]` becomes tiny
or non-positive in fp32 and the unguarded `1/d[j]` returns `Inf/NaN`, which
silently propagates through `gi` and every downstream `bmm`, corrupting the whole
forecast. CPU `torch.linalg.lstsq` degrades gracefully here (min-norm); the raw
inverse did not.

### The fix
Clamp every pivot to a small floor relative to the matrix scale:
`scale = max_j |g[j][j]|`, `pivotFloor = 1e-6 * scale`, `d[j] = max(d[j], pivotFloor)`.
This is a localized Tikhonov ridge applied only to deficient pivots.

### CPU verification (`tests/batch_spd_inv_cpu_ref.cpp`, no NPU needed)
A host oracle reproduces the kernel's exact fp32 LDL^T arithmetic and compares
against an independent fp64 Gauss-Jordan inverse.
- **Well-conditioned SPD (m=3..64):** kernel vs fp64 oracle max_diff <= 8.8e-7,
  residual `|g·gi − I|` <= 8.6e-7 (gate 1e-5). The guard is **bit-identical** to
  the unguarded kernel on these inputs (resid_kernel == resid_guard) — zero
  accuracy regression on the real Koopa workload.
- **Rank-deficient / collinear Gram:** unguarded kernel emits `NaN/Inf`
  (residual = inf); guarded kernel stays finite and bounded (residual ~1.0).

### Honest positioning
The guard is a **stability net**, not a min-norm pseudo-inverse: on a truly
singular input it returns a finite, bounded (ridge-regularized) result rather
than NaN — it does not reproduce lstsq's exact min-norm solution there. For the
full-rank Koopa workload the output is unchanged. The previously built artifacts
under `generated/` predate this change; the migrated source has now been rebuilt and validated on Ascend 910B3
(`build_batch_spd_inv_node.sh`) before re-validation.
