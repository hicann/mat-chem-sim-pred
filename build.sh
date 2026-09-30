#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# mat-chem-sim-pred 统一编译入口（替代 .gitcode/scripts/ascend_build_test.sh，参考 cann/ops-nn）
# CI 流水线统一调用本脚本，新增算子无需修改流水线。用法: bash build.sh --help
# ============================================================================
set -eo pipefail

# 默认参数
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
BUILD_TYPE="Release"
JOBS=$(nproc 2>/dev/null || echo 8)
RUN_TEST=false
BUILD_ALL=false
LIST_ONLY=false
CLEAN_MODE=false
SKIP_CHECK=false
BUILD_ONLY="${BUILD_ONLY:-false}"
FILELIST=""
OPS_BY_NAME=""
PYTHON_OP=""
COMMON_DIRS=""
SCOPE=""
REFERENCE_TEST=false
OPERATORS=()

# 环境变量
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"

# 锚定仓库根目录
cd "$(dirname "$0")"

# 算子根目录（新增模块时在此添加）
OPERATOR_ROOTS=(
    "simulation/AI4MD"
    "simulation/AI4PDE"
    "prediction/ProcessControl/PIDModelFit"
    "prediction/ProcessControl/TimeSeriesForecast"
    "prediction/ProcessControl/NonTemporalPrediction"
)

# 共享目录（DirectMatchDir）：只认直接文件变更（不含子目录）触发全量重建
DIRECT_MATCH_DIRS=(
    "prediction/ProcessControl/NonTemporalPrediction/"
    "prediction/ProcessControl/ContinuousPrediction/"
    "prediction/ProcessControl/DiscretePrediction/"
)

# 动态发现含 CMakeLists.txt 的算子
ALL_OPERATORS=()
for root in "${OPERATOR_ROOTS[@]}"; do
    [ -d "$root" ] || continue
    for d in "$root"/*/; do
        if [ -f "${d}CMakeLists.txt" ]; then
            ALL_OPERATORS+=("${d%/}")
        fi
    done
done

# Python 算子（无 CMakeLists.txt，走 pytest）
PYTHON_OPERATORS=(
    "simulation/MaterialPropertyPrediction/DAO"
    "prediction/SmallData"
)

# 帮助信息
usage() {
    cat <<'EOF'
mat-chem-sim-pred Build Script

Usage:
  bash build.sh <operator>...            Build specified operators (path or name)
  bash build.sh --all                    Build all discovered operators
  bash build.sh --list                   List all discovered operators
  bash build.sh --clean [operator]...    Clean build artifacts
  bash build.sh --test [operator]...     Build and run ctest
  bash build.sh -f <filelist>            Incremental build from file list
  bash build.sh --ops=op1,op2,...        Build by operator names
  bash build.sh --python <operator>      Python operator pytest test
  bash build.sh --reference-test [prefix] Run pytest reference tests for operators

Options:
  --soc=<SOC>        SOC version (default: Ascend910B1)
  --debug            Debug build type (CMAKE_BUILD_TYPE=Debug)
  -O<level>          Optimization level (O0/O1/O2/O3, maps to Release)
  -j<N>              Parallel build jobs (default: nproc)
  -f <filelist>      File list for incremental build (one path per line)
  --ops=<ops>        Operator names, comma-separated
  --scope=<prefix>   Path prefix filter (-f mode only); common-dir changes only
                     rebuild operators under this prefix, not the whole repo
  --common-dirs=<dirs>  Shared dirs (space-separated); changes trigger full rebuild
  --python <op>      Python operator test via pytest (auto-detect test_ops.py/test_gpr.py/tests/)
  --reference-test   Run pytest reference tests for CMake operators with tests/ dirs
  --build-only        Build only, skip ctest
  --skip-check        Skip "md/tests-only" check for -f mode
  -h, --help         Show this help

Examples:
  bash build.sh simulation/AI4MD/Lennard_Jones
  bash build.sh Lennard_Jones pinn
  bash build.sh --ops=Lennard_Jones,pinn --test
  bash build.sh -f pr_filelist.txt
  bash build.sh -f pr_filelist.txt --test --common-dirs="prediction/ProcessControl/PIDModelFit/common/"
  bash build.sh --all -j16
  bash build.sh --python simulation/MaterialPropertyPrediction/DAO -f pr_filelist.txt
  bash build.sh --python prediction/SmallData -f pr_filelist.txt
  bash build.sh --reference-test prediction/ProcessControl/ -f pr_filelist.txt
  bash build.sh --clean --all

Note:
  - Python operators (DAO, GPR) use --python; they are excluded from CMake build.
  - --common-dirs: changes trigger full rebuild of the batch; for DirectMatchDirs
    (NonTemporalPrediction/ etc.) only direct file changes count.
EOF
}

# Source CANN 环境
source_cann_env() {
    local env_script=""
    for path in \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "${ASCEND_HOME_PATH}/set_env.sh" \
        "${ASCEND_TOOLKIT_HOME}/set_env.sh"; do
        if [ -f "$path" ]; then
            env_script="$path"
            break
        fi
    done
    if [ -n "$env_script" ]; then
        source "$env_script" 2>/dev/null || true
    fi
}

# 算子名 → 算子路径映射
resolve_operator() {
    local input="$1"
    if [ -f "$input/CMakeLists.txt" ]; then
        echo "$input"
        return 0
    fi
    local basename="${input##*/}"
    for op in "${ALL_OPERATORS[@]}"; do
        if [ "${op##*/}" = "$basename" ]; then
            echo "$op"
            return 0
        fi
    done
    return 1
}

# 判断目录是否为 DirectMatchDir
is_direct_match_dir() {
    local cdir="$1"
    for d in "${DIRECT_MATCH_DIRS[@]}"; do
        if [ "$cdir" = "$d" ]; then
            return 0
        fi
    done
    return 1
}

# 从文件列表提取受影响算子（去重）；共享目录有变更时输出 ALL 标记
resolve_operators_from_filelist() {
    local filelist="$1"
    local common_dirs_str="$2"
    local -a result=()
    local -A seen=()
    local common_changed=false

    # 检查 COMMON_DIRS 是否有变更
    if [ -n "$common_dirs_str" ]; then
        for cdir in $common_dirs_str; do
            if is_direct_match_dir "$cdir"; then
                # DirectMatchDir: 只匹配直接文件变更（非子目录）
                if grep -qE "^${cdir}[^/]+$" "$filelist" 2>/dev/null; then
                    common_changed=true
                    break
                fi
            else
                # 普通目录: 前缀匹配
                if grep -q "^${cdir}" "$filelist" 2>/dev/null; then
                    common_changed=true
                    break
                fi
            fi
        done
    fi

    if [ "$common_changed" = true ]; then
        echo "ALL"
        return 0
    fi

    # 逐行匹配算子目录（|| [ -n "$line" ] 兜底无换行结尾的最后一行）
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"  # 兼容 CRLF 行尾
        [ -z "$line" ] && continue
        [[ "$line" =~ ^# ]] && continue

        for op in "${ALL_OPERATORS[@]}"; do
            if [[ "$line" == "$op/"* ]]; then
                if [ -z "${seen[$op]}" ]; then
                    seen[$op]=1
                    result+=("$op")
                fi
                break
            fi
        done
    done < "$filelist"

    printf '%s\n' "${result[@]}"
}

# 检查文件列表是否只有 .md 或 tests/ 文件
should_skip_build() {
    local filelist="$1"
    local non_skip_count
    # 兼容 CRLF；grep -c 计数为 0 时退出码为 1，用 || true 兜底
    non_skip_count=$(tr -d '\r' < "$filelist" 2>/dev/null | grep -vE '(\.md$|^tests/|/tests/)' | grep -cv '^$' || true)
    non_skip_count=${non_skip_count:-0}
    [ "$non_skip_count" -eq 0 ]
}

# 编译单个 CMake 算子
build_single() {
    local op="$1"
    if [ ! -f "$op/CMakeLists.txt" ]; then
        echo "[SKIP] $op — CMakeLists.txt not found"
        return 1
    fi
    echo "[BUILD] $op (SOC=$SOC_VERSION, type=$BUILD_TYPE, jobs=$JOBS)"

    cmake -S "$op" -B "$op/build" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DSOC_VERSION="$SOC_VERSION" \
        || { echo "[FAIL] $op — cmake configure failed"; return 1; }

    ( cd "$op/build" && make -j"$JOBS" ) \
        || { echo "[FAIL] $op — compile failed"; return 1; }

    if [ "$BUILD_ONLY" != "true" ] && [ "$RUN_TEST" = true ] && [ -d "$op/tests" ]; then
        echo "[TEST] $op"
        ( cd "$op/build" && ctest --output-on-failure -j"$JOBS" ) \
            || { echo "[FAIL] $op — ctest failed"; return 1; }
    fi
    echo "[ OK ] $op"
    return 0
}

# Python 算子测试：自动检测 test_ops.py / test_gpr.py / tests/
run_python_test() {
    local op="$1"
    local filelist="$2"

    # 增量检测：filelist 中该算子无变更则跳过
    if [ -n "$filelist" ] && [ -f "$filelist" ]; then
        if ! grep -q "^${op}/" "$filelist" 2>/dev/null; then
            echo "[SKIP] No ${op} files changed; Python tests are skipped."
            return 0
        fi
    fi

    echo "[PYTEST] $op"
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh" 2>/dev/null || true
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    "$PYTHON_BIN" -m pip install pytest numpy 2>/dev/null || true
    "$PYTHON_BIN" - <<'PY'
import importlib.util, subprocess, sys
required = ["numpy", "pytest", "torch"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

    if [ -f "${op}/test_ops.py" ]; then
        "$PYTHON_BIN" -X faulthandler -m pytest "${op}/test_ops.py" -v --tb=short
    elif [ -f "${op}/test_gpr.py" ]; then
        "$PYTHON_BIN" -X faulthandler -m pytest "${op}/test_gpr.py" -v --tb=short
    elif [ -d "${op}/tests" ]; then
        ( cd "$op" && "$PYTHON_BIN" -X faulthandler -m pytest tests -v --tb=short )
    else
        echo "[WARN] No Python test file found in $op"
        return 1
    fi
    return $?
}

# 参考测试（stage3）：遍历 CMake 算子的 tests/ 目录运行 pytest
run_reference_tests() {
    local filelist="$1"
    local filter_prefix="$2"  # 如 "prediction/ProcessControl/" 只测该前缀下的算子
    local common_dirs_str="$3"  # 共享目录，有变更时全量测试该批算子
    local overall_rc=0
    local ran_test=false
    local common_changed=false

    # 检查 COMMON_DIRS 是否有变更
    if [ -n "$common_dirs_str" ] && [ -n "$filelist" ] && [ -f "$filelist" ]; then
        for cdir in $common_dirs_str; do
            if is_direct_match_dir "$cdir"; then
                if grep -qE "^${cdir}[^/]+$" "$filelist" 2>/dev/null; then
                    common_changed=true
                    break
                fi
            else
                if grep -q "^${cdir}" "$filelist" 2>/dev/null; then
                    common_changed=true
                    break
                fi
            fi
        done
    fi

    source "${ASCEND_TOOLKIT_HOME}/set_env.sh" 2>/dev/null || true
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    "$PYTHON_BIN" -m pip install pytest numpy 2>/dev/null || true
    "$PYTHON_BIN" - <<'PY'
import importlib.util, subprocess, sys
required = ["numpy", "pytest", "torch"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

    for op in "${ALL_OPERATORS[@]}"; do
        # 前缀过滤
        if [ -n "$filter_prefix" ] && [[ "$op" != "$filter_prefix"* ]]; then
            continue
        fi
        # 必须有 tests/ 目录
        [ -d "$op/tests" ] || continue
        # 共享目录变更时已确定全量，否则按 filelist 过滤
        if [ "$common_changed" = false ] && [ -n "$filelist" ] && [ -f "$filelist" ]; then
            if ! grep -q "^${op}/" "$filelist" 2>/dev/null; then
                continue
            fi
        fi
        ran_test=true
        echo "=== Reference test: $op ==="
        if ! ( cd "$op" && "$PYTHON_BIN" -X faulthandler -m pytest tests -v --tb=short ); then
            overall_rc=1
        fi
    done

    if [ "$ran_test" = false ]; then
        echo "No operator files changed; reference tests are skipped."
    fi
    return $overall_rc
}

# 清理单个算子的构建产物
clean_single() {
    local op="$1"
    if [ -d "$op/build" ]; then
        echo "[CLEAN] $op/build"
        rm -rf "$op/build"
    fi
}

# 解析参数
while [ $# -gt 0 ]; do
    case "$1" in
        --all)          BUILD_ALL=true ;;
        --list)         LIST_ONLY=true ;;
        --test)         RUN_TEST=true ;;
        --clean)        CLEAN_MODE=true ;;
        --debug)        BUILD_TYPE="Debug" ;;
        --build-only)   BUILD_ONLY=true ;;
        --skip-check)   SKIP_CHECK=true ;;
        --soc=*)        SOC_VERSION="${1#--soc=}" ;;
        --ops=*)        OPS_BY_NAME="${1#--ops=}" ;;
        --scope=*)      SCOPE="${1#--scope=}" ;;
        --common-dirs=*) COMMON_DIRS="${1#--common-dirs=}" ;;
        --python)
            shift
            PYTHON_OP="${1:-}"
            [ -z "$PYTHON_OP" ] && { echo "ERROR: --python requires an operator path"; exit 1; }
            ;;
        --reference-test) REFERENCE_TEST=true ;;
        -f)
            shift
            FILELIST="${1:-}"
            [ -z "$FILELIST" ] && { echo "ERROR: -f requires a file list path"; exit 1; }
            ;;
        -O[0-3])        BUILD_TYPE="Release" ;;
        -j*)            JOBS="${1#-j}"; [ -z "$JOBS" ] && JOBS=8 ;;
        -h|--help)      usage; exit 0 ;;
        *)              OPERATORS+=("$1") ;;
    esac
    shift
done

# Source CANN 环境
source_cann_env

# 列出算子
if [ "$LIST_ONLY" = true ]; then
    echo "Discovered ${#ALL_OPERATORS[@]} CMake operators:"
    printf '  %s\n' "${ALL_OPERATORS[@]}"
    echo ""
    echo "Python operators (use --python):"
    printf '  %s\n' "${PYTHON_OPERATORS[@]}"
    exit 0
fi

# Python 算子模式
if [ -n "$PYTHON_OP" ]; then
    run_python_test "$PYTHON_OP" "$FILELIST"
    exit $?
fi

# 参考测试模式
if [ "$REFERENCE_TEST" = true ]; then
    # 位置参数可作前缀过滤（如 prediction/ProcessControl/）
    FILTER_PREFIX="${OPERATORS[0]:-}"
    run_reference_tests "$FILELIST" "$FILTER_PREFIX" "$COMMON_DIRS"
    exit $?
fi

# 处理 -f 模式：从文件列表增量编译
if [ -n "$FILELIST" ]; then
    if [ ! -f "$FILELIST" ]; then
        echo "ERROR: file list not found: $FILELIST"
        exit 1
    fi

    echo "[INFO] Reading file list: $FILELIST"

    # 跳过检查：仅 .md 或 tests/ 文件时跳过编译；但 --test 模式下仍跑 ctest
    if [ "$SKIP_CHECK" = false ]; then
        if should_skip_build "$FILELIST"; then
            if [ "$RUN_TEST" = true ]; then
                echo "[INFO] Only .md/tests files changed; build skipped, running tests only"
            else
                echo "[SKIP] File list contains only .md or tests/ files, skip build"
                mkdir -p build_out 2>/dev/null || true
                touch build_out/skip_build.run 2>/dev/null || true
                exit 0
            fi
        fi
    fi

    # 从文件列表提取受影响的算子（含 COMMON_DIRS 检查）
    mapfile -t FILELIST_RESULT < <(resolve_operators_from_filelist "$FILELIST" "$COMMON_DIRS")

    # ALL（共享目录变更）按 --scope 限定，只重建该批算子而非全仓
    if [ "${#FILELIST_RESULT[@]}" -gt 0 ] && [ "${FILELIST_RESULT[0]}" = "ALL" ]; then
        echo "[INFO] Common directory changed, building operators in scope: ${SCOPE:-<all>}"
        for op in "${ALL_OPERATORS[@]}"; do
            if [ -z "$SCOPE" ] || [[ "$op" == "$SCOPE"* ]]; then
                OPERATORS+=("$op")
            fi
        done
    else
        # 过滤空行并按 --scope 限定范围
        for op in "${FILELIST_RESULT[@]}"; do
            [ -z "$op" ] && continue
            if [ -z "$SCOPE" ] || [[ "$op" == "$SCOPE"* ]]; then
                OPERATORS+=("$op")
            fi
        done
    fi

    if [ ${#OPERATORS[@]} -eq 0 ]; then
        echo "[SKIP] No affected operators found in file list"
        mkdir -p build_out 2>/dev/null || true
        touch build_out/skip_build.run 2>/dev/null || true
        exit 0
    fi

    echo "[INFO] Affected operators: ${#OPERATORS[@]}"
    printf '  %s\n' "${OPERATORS[@]}"
fi

# 处理 --ops= 模式：按算子名编译
if [ -n "$OPS_BY_NAME" ]; then
    IFS=',' read -ra name_arr <<< "$OPS_BY_NAME"
    for name in "${name_arr[@]}"; do
        name="${name#"${name%%[![:space:]]*}"}"  # 去前导空格（不依赖 xargs）
        name="${name%"${name##*[![:space:]]}"}"   # 去尾部空格
        if ! resolved=$(resolve_operator "$name"); then
            echo "[WARN] Operator not found: $name"
            continue
        fi
        OPERATORS+=("$resolved")
    done
fi

# 确定操作列表
if [ "$BUILD_ALL" = true ] || ([ "$CLEAN_MODE" = true ] && [ ${#OPERATORS[@]} -eq 0 ]); then
    if [ ${#ALL_OPERATORS[@]} -eq 0 ]; then
        echo "ERROR: no operators discovered — check directory structure"
        exit 1
    fi
    OPERATORS=("${ALL_OPERATORS[@]}")
fi

# 无算子时显示帮助
if [ ${#OPERATORS[@]} -eq 0 ]; then
    usage
    exit 0
fi

# 清理模式
if [ "$CLEAN_MODE" = true ]; then
    for op in "${OPERATORS[@]}"; do
        clean_single "$op"
    done
    echo "----------------------------------------------------------------"
    echo "Clean done: ${#OPERATORS[@]} operator(s)."
    exit 0
fi

# 编译模式
failed=0
succeeded=0
for op in "${OPERATORS[@]}"; do
    if build_single "$op"; then
        succeeded=$((succeeded + 1))
    else
        failed=$((failed + 1))
    fi
done

echo "----------------------------------------------------------------"
echo "Done: $succeeded succeeded, $failed failed."
[ "$failed" -eq 0 ] || exit 1
