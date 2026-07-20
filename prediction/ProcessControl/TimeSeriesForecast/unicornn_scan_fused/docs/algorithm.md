# UnicornnScanFused — design notes

## Why this is a strong NPU custom-op case
UnICORNN (undamped independent controlled oscillatory RNN, Rusch & Mishra ICML'21,
same group as LEM / coRNN) is a Hamiltonian-inspired second-order oscillator network:
position `y` and velocity `z` evolve through a per-neuron, sigmoid-controlled step size.
Crucially the hidden-hidden recurrence is **diagonal** (`w (.) y`, independent per
channel) — only the input projection `V x_s` is a matmul. The tanh feedback makes the
state update nonlinear so there is **no parallel/chunked rewrite**, and there is **no
native NPU primitive**. Frameworks run it as a per-timestep Python loop of tiny
elementwise + small-matmul launches — almost pure launch-bound overhead.

Gap-hunt (Ascend 910B3): torch per-timestep loop vs single batched input-projection
GEMM floor measured a **2132× / 1005× launch-bound gap** (83.1ms vs 0.039ms at
B32 L336 IN11 H64; 23.5ms vs 0.023ms at B16 L96 IN7 H128). Because the diagonal
recurrence carries almost no FLOPs, the gap (and the realized fusion speedup) is the
largest of the batch-3 family. Same family as the delivered LEM / coRNN / CfC / TiRex
sLSTM / WITRAN strong cases.

## Mapping to 910B3
- **Batch-parallel**: each AI core owns a contiguous slab of batches; `block_dim =
  min(B, 32)`. States `y,z` live in UB for the whole time loop — no GM round-trip, no
  SyncAll, no workspace.
- **Diagonal recurrence**: `w (.) y` is a single elementwise `Mul` over length H (no
  matmul). The only contraction is the input projection `V x_s` (over `IN`), done via
  an axpy accumulation over the contraction axis (matches the transposed-weight
  convention of the GRU / TiRex / LEM / CfC / coRNN kernels). Stays on the vector core.
- **Control precomputed once**: `sig_c = sigmoid(c)` is time-constant, computed once
  per core before the time loop and reused across all timesteps/batches.
- **Activation synthesized**: no native Tanh → `tanh(x)=2*sigmoid(2x)-1`,
  `sigmoid(x)=1/(1+exp(-x))` (clamped ±80 before exp).
- **IMEX-style update on the vector core**: `z -= dt*sig_c*(tanh(arg)+alpha*y)` then
  `y += dt*sig_c*z`, all elementwise Muls/Mul/Add/Sub over length H.
- **Alignment**: feature load uses `DataCopyPad` with explicit byte length so
  non-8-multiple `IN` (e.g. 11, 21) does not over/under-read GM. Probe covers IN=21
  and IN=11; both pass ≤2.98e-8.

## Constants
`dt=0.042, alpha=1.0` are baked into the kernel and matched by the framework torch
baseline + CPU oracle (mirrors the `dt`-baked precedent in the LEM / coRNN kernels).
To re-tune, change the two constants in the kernel, oracle
(`unicornn_scan_fused_probe.cpp`) and the harness (`unicornn_e2e.py`) together.

## Weight format
`weight:[IN+2, H]` column-packed transposed: `[0:IN,:]=V` (input projection),
`[IN,:]=w` (diagonal recurrent weight), `[IN+1,:]=c` (per-neuron control). `bias:[H]=b`.
Output `[B,L,H]` (= the position state `y`).

## Three-gate results
Correctness probe 5.53e-10~2.98e-8; fused encoder vs fp32 torch 2.68e-7; component
160.68× (79.76ms→0.496ms); E2E 3-layer 100.43× (243.79ms→2.43ms). Verdict **STRONG**.
