# TirexSlstmCell — vectorized fused sLSTM scan (design)

Device: Ascend 910B3, CANN 8.1.RC1.
Operator: `TirexSlstmCell` (`aclnnTirexSlstmCell`).
Value class: **supported -> faster** (price point #2). Gated model: **NX-AI/TiRex**
(xLSTM, 2025) — a hot non-transformer time-series foundation model whose block0
is 100% sLSTM recurrence.

## 1. What it computes

One sLSTM cell forward over a whole sequence. Per head `r` (head_dim = `Hd`
lanes) and per timestep `t`, with pre-activations

```
raw_g = input[b,t,g,:] + bias[g,:] + (prev_h_head @ R[r])_g      g in {i,f,z,o}
```

the stabilized sLSTM recurrence (matching the TiRex reference exactly) is

```
logfplusm = prevM + logsigmoid(fraw)            # = prevM - ln(1+exp(-fraw))
mnew      = (t==0 && initial n all-zero) ? iraw : max(iraw, logfplusm)
igate     = min(exp(iraw - mnew), 1)
fgate     = min(exp(logfplusm - mnew), 1)
zgate     = tanh(zraw)
ogate     = sigmoid(oraw)
cnew      = fgate*prevC + igate*zgate
nnew      = fgate*prevN + igate
hnew      = ogate * cnew / (nnew==0 ? 1e-6 : nnew)
```

## 2. Interface (msopgen IR)

Inputs (all fp32 ND):
- `input`           `[B, S, 4*H]`  gate-major within the last axis, gate order **[i, f, z, o]**, hidden natural (`g*H + head*Hd + pos`). These are the **input projections only** (W_x·x); recurrent and bias are added inside the op.
- `recurrent_kernel` `[num_heads, Hd, 4*Hd]`  per-head, last axis gate-major `[i,f,z,o]` (`gate*Hd + pos`).
- `bias`            `[4*H]`  gate-major `[i,f,z,o]` (this is TiRex `cell._bias_` permuted `reshape(heads,gates,Hd).permute(1,0,2)`).
- `initial_state`   `[4, B, H]`  state order **[h, c, n, m]**.

Outputs (fp32 ND):
- `output`          `[B, S, H]`  the h sequence.
- `final_state`     `[4, B, H]`  final [h, c, n, m].

`H = num_heads * Hd`. TiRex profiling shape: B=64, S=64, H=512, heads=4, Hd=128.

## 3. Parallelization

Each `(batch, head)` pair is an **independent** sLSTM scan over `Hd` lanes, so
there are `groups = B * num_heads` independent scans (256 at the profiling
shape). They are spread across all AI vector cores grid-stride:

```
groups   = batch * num_heads
block_dim = min(groups, PlatformAscendC().GetCoreNumAiv())   # host tiling
```

Each core owns `count = groups/blockNum (+1 for the tail)` groups and walks them
grid-stride. The original kernel hard-coded `block_dim = min(batch, 32)`, which
left most vector cores idle and was the single biggest perf lever (see results).

## 4. Per-step compute (all on the vector unit)

For a group's `Hd`-lane:

1. **Recurrent contribution** `acc[4*Hd] = prev_h_head @ R[head]`: computed as
   `Hd` row-axpys (`acc += prev_h[j] * R[j, :]`), with R rows streamed in blocks
   of `kRowBlk = 8` via `DataCopy`. R (1 MB) stays L2-resident and is reused
   every step. (`kRowBlk = 16` was tried and is marginally slower — the matvec
   is not DataCopy-count bound.)
2. **Raw gates** `raw = input + bias + acc` (`Adds` to stage input/bias slices,
   then two `Add`s).
3. **Nonlinearities**: this CANN build has no native `Tanh`/`Sigmoid`, so they
   are synthesized from `Exp`/`Ln`:
   - `sigmoid(x) = 1/(1+exp(-x))` (input clamped to ±80 for fp32 safety),
   - `tanh(x)   = 2*sigmoid(2x) - 1`,
   - `logsigmoid(x) = -ln(1+exp(-x))`.
4. **State update** `cnew/nnew/hnew` via `Mul`/`Add`/`Div`. Buffers are aliased
   in place (`cnew` over `rawI`, `nnew` over `rawF`, `hnew` over `rawZ`,
   `ogate` over `rawO`) to keep UB footprint small.
5. **Writeback**: `h -> output[b,t,:]` and `final[0]`; `c/n/m -> final[1/2/3]`.

State layout reuses the op's own GM buffers exactly like the original scalar
kernel: `h_{t-1}` is read back from `outputGm`, `c/n/m` live in `finalStateGm`
(init-copied from `initial_state` in an `InitGroup` pre-pass). This keeps the
op/probe interface unchanged — it is a drop-in replacement.

## 5. Synchronization

The step loop is the **outer** loop and is identical across all cores, so a
single `SyncAll()` per timestep gives uniform participation (no deadlock) and
orders each step's GM writes before the next step's `prev_h` reads. There is no
hand-written `HardEvent` (the repo has zero precedent for it); only the
established `TQue` DeQue + `SyncAll` synchronization is used.

## 6. Why this wins

torch's `sLSTMLayer.cell_impl` runs the scan as 64 sequential timestep launches,
each doing pointwise sigmoid/tanh/exp + state update; the profiled cost is
dominated by the pointwise gates (46.2 ms of 54.6 ms = 88%), not the recurrent
matmul (9.9 ms). Fusing the entire scan into one launch with vectorized gates,
across all AI cores, removes the per-step launch overhead — the same structural
win as `SelectiveScan1D` (~10×). See `docs/*results.json` for measured numbers.
