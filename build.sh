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

# ---------------------------------------------------------------------------
# 新入口（迁移期双轨）：只读子命令转交 scripts/build_main.sh，
# 其余参数仍走本文件原有逻辑。子命令逐个搬迁，搬完再删旧实现。
# ---------------------------------------------------------------------------
for _arg in "$@"; do
    case "$_arg" in
        --list | --list-ops | --help | -h | -f | --test | --build-only | --all | --ops=* | \
            --python | --python-tests | --python-tests=* | --reference-test)
            REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
            exec bash "$REPO_ROOT/scripts/build_main.sh" "$@" ;;
    esac
done
unset _arg

# 默认参数
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
BUILD_TYPE="Release"
JOBS=$(nproc 2>/dev/null || echo 8)
RUN_TEST=false
HAS_GTEST=false
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

# 暂不纳入 CI 构建的算子：实现尚未完成，补齐后从此处移除
# - simulation/AI4MD/Dissipative_particle_dynamics: dpd_host.cpp 实现被截断，
#   run_single_step/free_device_memory 与公开 API 均未实现
SKIP_OPERATORS=(
    "simulation/AI4MD/Dissipative_particle_dynamics"
)

# 动态发现含 CMakeLists.txt 的算子
ALL_OPERATORS=()
for root in "${OPERATOR_ROOTS[@]}"; do
    [ -d "$root" ] || continue
    for d in "$root"/*/; do
        if [ -f "${d}CMakeLists.txt" ]; then
            op="${d%/}"
            skip=false
            for s in "${SKIP_OPERATORS[@]}"; do
                if [ "$op" = "$s" ]; then
                    skip=true
                    break
                fi
            done
            if [ "$skip" = false ]; then
                ALL_OPERATORS+=("$op")
            fi
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

# Python 算子测试：自动检测 test_ops.py / test_gpr.py / tests/
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
    if [ ${#SKIP_OPERATORS[@]} -gt 0 ]; then
        echo ""
        echo "Excluded from build (WIP):"
        printf '  %s\n' "${SKIP_OPERATORS[@]}"
    fi
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
