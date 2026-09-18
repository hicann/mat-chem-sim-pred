#!/usr/bin/env bash
# ============================================================================
# Ascend 算子编译与测试脚本（stage4 各 job 共用）
#
# 用法:
#   ascend_build_test.sh single <operator_dir>
#   ascend_build_test.sh loop_fixed "<dir1> <dir2> ..."
#   ascend_build_test.sh loop_glob "<glob_pattern>" ["<common_dirs>"]
#   ascend_build_test.sh python <operator_dir>
#
# 变更过滤: 读取仓库根目录的 pr_filelist.txt（由流水线 obs-download 步骤下载），
# 只有路径命中的算子才会被编译/测试。
# ============================================================================
set -euo pipefail

MODE="${1:?usage: ascend_build_test.sh <single|loop_fixed|loop_glob|python> <args...>}"
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"
BUILD_ONLY="${BUILD_ONLY:-false}"

changed_files="$(cat pr_filelist.txt)"

run_single() {
    local operator="$1"
    echo "=== cmake configure: $operator ==="
    cmake -S "$operator" -B "$operator/build" -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1 || return 1
    echo "=== cmake build: $operator ==="
    cmake --build "$operator/build" -j"$(nproc)" || return 1
    if [ "$BUILD_ONLY" != 'true' ] && [ -d "$operator/tests" ]; then
        echo "=== ctest: $operator ==="
        ctest --test-dir "$operator/build" --output-on-failure -j"$(nproc)" || return 1
    fi
    return 0
}

case "$MODE" in

single)
    OPERATOR="$1"
    if ! printf '%s\n' "$changed_files" | grep -q "^${OPERATOR}/"; then
        echo "No ${OPERATOR} files changed; build and tests are skipped."
        exit 0
    fi
    test -f "$OPERATOR/CMakeLists.txt"
    run_single "$OPERATOR"
    ;;

loop_fixed)
    ran_build=false
    overall_rc=0
    for operator in "$@"; do
        if ! printf '%s\n' "$changed_files" | grep -q "^${operator}/"; then
            continue
        fi
        ran_build=true
        if ! run_single "$operator"; then
            overall_rc=1
        fi
    done
    if [ "$ran_build" = false ]; then
        echo "No operator files changed; build is skipped."
    fi
    exit "$overall_rc"
    ;;

loop_glob)
    GLOB_PATTERN="$1"
    COMMON_DIRS="${2:-}"
    common_changed=false
    if [ -n "$COMMON_DIRS" ]; then
        for cdir in $COMMON_DIRS; do
            if printf '%s\n' "$changed_files" | grep -q "^${cdir}"; then
                common_changed=true
                break
            fi
        done
    fi
    ran_test=false
    overall_rc=0
    for operator in $GLOB_PATTERN; do
        [ -d "$operator" ] && [ -f "$operator/CMakeLists.txt" ] || continue
        run_operator=false
        if printf '%s\n' "$changed_files" | grep -q "^${operator}/"; then
            run_operator=true
        fi
        [ "$common_changed" = true ] && run_operator=true
        if [ "$run_operator" = false ]; then
            continue
        fi
        ran_test=true
        if ! run_single "$operator"; then
            overall_rc=1
        fi
    done
    if [ "$ran_test" = false ]; then
        echo "No operator files changed; build and tests are skipped."
    fi
    exit "$overall_rc"
    ;;

python)
    OPERATOR="$1"
    if ! printf '%s\n' "$changed_files" | grep -q "^${OPERATOR}/"; then
        echo "No ${OPERATOR} files changed; tests are skipped."
        exit 0
    fi
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
    python3 -m pip install pytest numpy
    python3 - <<'PY'
import torch
assert torch.npu.is_available(), "PyTorch NPU is not available"
PY
    python3 -m pytest "${OPERATOR}/test_ops.py" -v --tb=short
    ;;

*)
    echo "ERROR: Unknown mode '$MODE'. Valid: single | loop_fixed | loop_glob | python"
    exit 1
    ;;

esac
