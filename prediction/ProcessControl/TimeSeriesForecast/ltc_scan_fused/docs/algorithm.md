# LtcScanFused — design notes

## Why this is a strong NPU custom-op case
The Liquid Time-Constant network (Hasani et al. AAAI'21) is a continuous-time RNN whose
per-neuron time constant is itself input-dependent. It is integrated with a *fused*
semi-implicit (closed-form) ODE solver that performs `K` inner unfolds per timestep:
```
h <- (cm/dt * h + gleak*Eleak + f*A) / (cm/dt + gleak + f),   f = sigmoid(W_rec h + W_in x + b)
```
Each unfold recomputes the recurrent matmul `W_rec h` and an elementwise division. The
sigmoid gate makes the update nonlinear → there is **no parallel/chunked rewrite**, and
there is **no native NPU primitive**. Frameworks run it as `K·L` tiny matmul + activation
+ division launches per sequence — the most launch-bound member of the batch-3 family.

Gap-hunt (Ascend 910B3): torch per-timestep loop vs a single batched hidden-hidden GEMM
floor (all `B·L·K` invocations collapsed) measured a **13992× / 4742× launch-bound gap**
(390.9ms vs 0.028ms at B32 L336 IN11 H64; 116.9ms vs 0.025ms at B16 L96 IN7 H128). Same
family as the delivered LEM / coRNN / UnICORNN / Lipschitz / CfC / TiRex sLSTM / WITRAN
strong cases. The `K=6` inner unfolds are what push the gap above every prior case.

## Mapping to 910B3
- **Batch-parallel**: each AI core owns a contiguous slab of batches; `block_dim =
  min(B, 32)`. State `h` lives in UB for the whole time loop — no GM round-trip, no
  SyncAll, no workspace.
- **Input projection hoisted**: `wx = W_in x_s` is computed once per timestep (constant
  across the K unfolds), so only the recurrent matmul `W_rec h` is recomputed per unfold.
- **Per-neuron constants precomputed once per core**: `cm/dt`, `gleak*Eleak`, and the
  denominator constant `cm/dt + gleak` are computed before the time loop and reused for
  all timesteps/unfolds/batches.
- **Projection on the vector core**: both `W_rec h` (over H) and `W_in x_s` (over IN) use
  an axpy accumulation over the contraction axis (matches the GRU / TiRex / LEM / CfC /
  coRNN convention), avoiding K=1 Cube degradation.
- **Activation synthesized**: no native sigmoid → `sigmoid(x)=1/(1+exp(-x))` (clamped
  ±80 before exp). The closed-form update uses an elementwise `Div`.
- **Alignment**: feature load uses `DataCopyPad` with explicit byte length so
  non-8-multiple `IN` (e.g. 11, 21) does not over/under-read GM. Probe covers IN=21 and
  IN=11; both pass ≤3.92e-6.

## Constants
`dt=0.042, K=6` are baked into the kernel and matched by the framework torch baseline +
CPU oracle (mirrors the `dt`-baked precedent in the LEM / coRNN / UnICORNN / Lipschitz
kernels). To re-tune, change the two constants in the kernel, oracle
(`ltc_scan_fused_probe.cpp`) and the harness (`ltc_e2e.py`) together. `gleak` and `cm`
must stay positive so the denominator `cm/dt + gleak + f` is well-conditioned (sigmoid
`f` is already in (0,1)); the probe/harness sample them positive.

## Weight / bias format
`weight:[H+IN, H]` column-packed transposed: `[0:H,:]=W_rec`, `[H:H+IN,:]=W_in`.
`bias:[5H]` packed per-neuron: `[b | gleak | Eleak | cm | A]`. Output `[B,L,H]` (= `h`).

## Three-gate results
Correctness probe 1.97e-6~3.92e-6; fused encoder vs fp32 torch 2.53e-7; component
82.44× (379.21ms→4.60ms); E2E 3-layer 76.90× (1134.71ms→14.76ms). Verdict **STRONG**.
