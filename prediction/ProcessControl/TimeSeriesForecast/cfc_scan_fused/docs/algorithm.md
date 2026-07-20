# CfcScanFused — design notes

## Why this is a strong NPU custom-op case
CfC (Closed-form Continuous-time, Hasani et al. 2022) is a liquid-time-constant RNN.
Its closed-form update couples two tanh candidates with a sigmoid time-interpolation
gate, with the state `h` fed back through nonlinearities every timestep. There is **no
parallel/chunked rewrite** (unlike linear-state SSMs: Mamba-2, RetNet, GLA, mLSTM),
and **no native NPU primitive**. Frameworks therefore run it as a per-timestep Python
loop of tiny matmul+activation launches — pure launch-bound overhead.

Gap-hunt (Ascend 910B3): torch per-timestep loop vs single batched-projection GEMM
floor measured a **2438× / 1088× launch-bound gap** (90.2ms vs 0.037ms at B32 L336;
24.6ms vs 0.023ms at B16 L96). This is the same family as the already-delivered LEM /
TiRex sLSTM / WITRAN strong cases.

## Mapping to 910B3
- **Batch-parallel**: each AI core owns a contiguous slab of batches; `block_dim =
  min(B, 32)`. State `h` lives in UB for the whole time loop — no GM round-trip, no
  SyncAll, no workspace (CfC has a single state, simpler than LEM's two states).
- **One projection pass**: `z=[h;x_s]` is projected once over all `3H` columns
  (Wf1|Wf2|Wt) via an axpy accumulation over the `H+IN` contraction axis (matches the
  transposed-weight convention of the GRU / TiRex / LEM kernels). Avoids K=1 Cube
  degradation by staying on the vector core.
- **Activations synthesized**: no native Tanh/Sigmoid → `sigmoid(x)=1/(1+exp(-x))`
  (clamped to ±80 before exp), `tanh(x)=2*sigmoid(2x)-1`.
- **Alignment**: feature load uses `DataCopyPad` with explicit byte length so
  non-8-multiple `IN` (e.g. 11, 21) does not over/under-read GM (the same 32B-alignment
  fix proven on WITRAN/LEM). Probe covers IN=21 and IN=11; both pass ≤2.3e-7.

## Weight format
`weight:[H+IN, 3H]` column-packed transposed: `[:,0:H]=Wf1`, `[:,H:2H]=Wf2`,
`[:,2H:3H]=Wt`. `bias:[3H]=concat(bf1,bf2,bt)`. Output `[B,L,H]`.

## Three-gate results
Correctness probe 9.87e-8~2.38e-7; fused encoder vs fp32 torch 2.33e-7; component
84.83× (104.39ms→1.23ms); E2E 3-layer 70.26× (330.0ms→4.70ms). Verdict **STRONG**.
