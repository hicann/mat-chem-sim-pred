# PID FOPDT full-chain E2E harness

End-to-end validation of the FOPDT PID-tuning pipeline, chaining the real
operators **fit → tuning_rule → fopdt_rollout → performance_metrics** and
comparing against a CPU reference.

Two tools are provided:

| Tool | Purpose | Compares against |
|---|---|---|
| `e2e_orchestrator.py` | **Accuracy**: drives the 4 operators stage-by-stage (`e2e_runner`) and checks each stage against its Python reference. | per-stage CPU reference (`common/*_reference.py`) |
| `e2e_perf` | **Performance**: single-process, device-resident chain `tuning_rule → fopdt_rollout → performance_metrics`, timed vs a CPU 64-thread chain; also re-checks final best-PID / score / metrics alignment. | CPU multi-thread chain (in-process) |

The rollout stage dominates the chain cost (tuning/metrics are ~0.05 ms each),
so the chain speedup tracks the rollout speedup.

## Build

The operators must be built first (each `<op>/build/lib<op>_host.so` and
`<op>/build/lib/lib<op>_kernel_lib.so` present). Then, from this directory:

```bash
bash build_e2e.sh        # produces ./e2e_perf and ./e2e_runner
```

Override the toolkit location with `ASCEND_HOME` / `ASCEND_TOOLKIT_ENV` if it is
not at the default `/usr/local/Ascend/ascend-toolkit/latest`.

## Run — performance (`e2e_perf`)

```bash
# args: <device> [batch=128] [candidates=1024] [sim_steps=1024] \
#       [candidate_tile=0:auto] [iters=5] [warmup=2] [threads=64]
./e2e_perf 0 128 16384 1024 0 5 2 64
```

`candidate_tile=0` lets the rollout operator auto-select the optimal tile
(`min(candidates, kLane=768)`); pass an explicit value only to sweep the knob.
Example representative-scale result (Ascend910B3, B=128, sim_steps=1024,
auto tile): C=1024 ≈ 4.0x, C=4096 ≈ 6.2x, C=16384 ≈ 4.5x vs CPU 64T.

## Run — accuracy (`e2e_orchestrator.py`)

```bash
export E2E_RUNNER=$PWD/e2e_runner   # required: path to the built runner
export E2E_WORK=/tmp/e2e_work       # optional: scratch dir for .bin I/O
# export PID_COMMON=/path/to/PIDModelFit/common   # optional override; defaults to ../common
python3 e2e_orchestrator.py
```

It writes a per-stage comparison report to `$E2E_WORK/e2e_report.json` and prints
the max error of each stage (NPU vs reference). All four stages align to within
float32 tolerance.
