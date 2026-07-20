# CornnScanFused — design notes

## Why this is a strong NPU custom-op case
coRNN (coupled oscillatory RNN, Rusch & Mishra ICLR'21, same group as LEM) is a
second-order nonlinear oscillator network: position `y` and velocity `z` are coupled
through a single `tanh` projection every timestep with an IMEX integration step.
There is **no parallel/chunked rewrite** (the tanh-coupled state update is not a
linear scan), and **no native NPU primitive**. Frameworks run it as a per-timestep
Python loop of tiny matmul+activation launches — pure launch-bound overhead.

Gap-hunt (Ascend 910B3): torch per-timestep loop vs single batched-projection GEMM
floor measured a **3310× / 784× launch-bound gap** (103.1ms vs 0.031ms at B32 L336;
28.6ms vs 0.036ms at B16 L96) — the largest of the batch-3 family so far. Same family
as the delivered LEM / TiRex sLSTM / WITRAN / CfC strong cases.

## Mapping to 910B3
- **Batch-parallel**: each AI core owns a contiguous slab of batches; `block_dim =
  min(B, 32)`. States `y,z` live in UB for the whole time loop — no GM round-trip, no
  SyncAll, no workspace.
- **One projection pass**: `zcat=[y;z;x_s]` (len `2H+IN`) is projected once to `H` via
  an axpy accumulation over the contraction axis (matches the transposed-weight
  convention of the GRU / TiRex / LEM / CfC kernels). Stays on the vector core,
  avoiding K=1 Cube degradation.
- **Activation synthesized**: no native Tanh → `tanh(x)=2*sigmoid(2x)-1`,
  `sigmoid(x)=1/(1+exp(-x))` (clamped ±80 before exp).
- **IMEX update on the vector core**: `z += dt*(tanh(pre) - gamma*y - eps*z)` then
  `y += dt*z`, all elementwise Muls/Add over length H.
- **Alignment**: feature load uses `DataCopyPad` with explicit byte length so
  non-8-multiple `IN` (e.g. 11, 21) does not over/under-read GM. Probe covers IN=21
  and IN=11; both pass ≤3.4e-8.

## Constants
`dt=0.042, gamma=1.0, eps=1.0` are baked into the kernel and matched by the framework
torch baseline + CPU oracle (mirrors the `dt`-baked precedent in the LEM kernel). To
re-tune, change the three constants in the kernel, oracle (`cornn_scan_fused_probe.cpp`)
and the harness (`cornn_e2e.py`) together.

## Weight format
`weight:[2H+IN, H]` column-packed transposed: `[0:H,:]=Wy`, `[H:2H,:]=Wz`,
`[2H:2H+IN,:]=V`. `bias:[H]`. Output `[B,L,H]` (= the position state `y`).

## Three-gate results
Correctness probe 2.33e-9~3.35e-8; fused encoder vs fp32 torch 1.49e-7; component
56.73× (97.96ms→1.73ms); E2E 3-layer 54.15× (338.73ms→6.26ms). Verdict **STRONG**.
