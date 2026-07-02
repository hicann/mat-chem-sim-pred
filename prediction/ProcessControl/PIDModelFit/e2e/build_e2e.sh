#!/usr/bin/env bash
# Build the FOPDT full-chain E2E harness (runner + perf) against the prebuilt
# PIDModelFit operator host .so libraries. Run from the e2e/ directory after the
# operators have been built (each <op>/build/lib<op>_host.so and
# <op>/build/lib/lib<op>_kernel_lib.so must exist).
set -euo pipefail
ASCEND="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}" >/dev/null 2>&1 || true

PERF_OPS=(pid_tuning_rule_batch pid_fopdt_batch_rollout_score pid_control_performance_metrics)
RUNNER_OPS=(pid_fopdt_basis_gemm_fit pid_tuning_rule_batch pid_fopdt_batch_rollout_score pid_control_performance_metrics)

libflags() {
  local flags=""
  for op in "$@"; do
    local bd="$ROOT/$op/build"
    flags="$flags -L$bd -l${op}_host -L$bd/lib -l${op}_kernel_lib"
    flags="$flags -Wl,-rpath,$bd -Wl,-rpath,$bd/lib -Wl,-rpath-link,$bd -Wl,-rpath-link,$bd/lib"
  done
  echo "$flags"
}

COMMON_FLAGS="-O2 -std=c++17 -I$ASCEND/include -L$ASCEND/lib64 -lascendcl -Wl,-rpath,$ASCEND/lib64"

echo "[build] e2e_perf"
g++ $COMMON_FLAGS "$HERE/e2e_perf.cpp" -o "$HERE/e2e_perf" $(libflags "${PERF_OPS[@]}")
echo "[build] e2e_runner"
g++ $COMMON_FLAGS "$HERE/e2e_runner.cpp" -o "$HERE/e2e_runner" $(libflags "${RUNNER_OPS[@]}")
echo "[build] done -> $HERE/e2e_perf  $HERE/e2e_runner"
