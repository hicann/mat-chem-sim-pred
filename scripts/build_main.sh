#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 统一入口的新实现（迁移期由 build.sh 分发进来）。
#
# 已接管：--list / --help / --python <op> / --python-tests[=cpu|npu|all]
# 未接管：--build / --test / --reference-test / --package 等仍走 build.sh 的旧实现，
# 随后逐个搬迁，每搬一个就删掉旧代码对应的一段。
# 配置的唯一来源仍是 cmake/variables.cmake（经 scripts/lib/config.sh 解析）。

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
cd "$REPO_ROOT"

for _lib in \
    lib/common.sh lib/config.sh lib/ops.sh \
    build_options.sh build_single.sh build_python.sh build_reference.sh; do
    # shellcheck source=/dev/null
    source "$REPO_ROOT/scripts/$_lib"
done
unset _lib

export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"

usage() {
    cat <<'EOF'
用法: bash build.sh [选项]

已由新入口接管：
  --list [--backend=ascendc|pypto|all]   列出参与构建的算子
  --python <operator>                    Python 算子 pytest
  --python-tests[=cpu|npu|all]           跑受影响的 Python 与 reference 测试
  -f <filelist> [--test] [--scope=前缀]   从变更清单增量编译（受 scope 限定）
  --ops=a,b                              按算子名编译
  --all                                  编译全部算子
  --help                                 显示本帮助

其余选项（--build/--test/--reference-test/--package 等）仍由 build.sh 的旧实现处理。
EOF
}

list_operators() {
    echo "Discovered ${#ALL_OPERATORS[@]} CMake operators:"
    printf '  %s\n' "${ALL_OPERATORS[@]}"
    echo ""
    echo "Python operators (use --python):"
    printf '  %s\n' "${PYTHON_OPERATORS[@]:-}"
    if [ ${#SKIP_OPERATORS[@]} -gt 0 ]; then
        echo ""
        echo "Excluded from build (WIP):"
        printf '  %s\n' "${SKIP_OPERATORS[@]}"
    fi
}

run_build() {
    local resolved op failed=0 succeeded=0
    if [ -n "$FILELIST" ]; then
        [ -f "$FILELIST" ] || die "file list not found: $FILELIST"
        echo "[INFO] Reading file list: $FILELIST"
        if [ "$SKIP_CHECK" = false ] && should_skip_docs_only "$FILELIST"; then
            echo "[SKIP] File list contains only .md files, skip build and test"
            mkdir -p build_out 2>/dev/null || true
            touch build_out/skip_build.run 2>/dev/null || true
            return 0
        fi
        local -a filelist_result=()
        while IFS= read -r resolved; do
            [ -z "$resolved" ] && continue
            filelist_result+=("$resolved")
        done < <(resolve_operators_from_filelist "$FILELIST" "$COMMON_DIRS")

        if [ "${#filelist_result[@]}" -gt 0 ] && [ "${filelist_result[0]}" = "ALL" ]; then
            echo "[INFO] Common directory changed, building operators in scope: ${SCOPE:-<all>}"
            for op in "${ALL_OPERATORS[@]}"; do
                matches_prefix "$op" "$SCOPE" && OPERATORS+=("$op")
            done
        else
            for op in "${filelist_result[@]}"; do
                matches_prefix "$op" "$SCOPE" && OPERATORS+=("$op")
            done
        fi

        if [ ${#OPERATORS[@]} -eq 0 ]; then
            echo "[SKIP] No affected operators found in file list"
            mkdir -p build_out 2>/dev/null || true
            touch build_out/skip_build.run 2>/dev/null || true
            return 0
        fi
        echo "[INFO] Affected operators: ${#OPERATORS[@]}"
        printf '  %s\n' "${OPERATORS[@]}"
    fi

    if gtest_available; then HAS_GTEST=true; fi
    if [ "$RUN_TEST" = true ] && [ "$HAS_GTEST" != true ]; then
        for op in "${OPERATORS[@]}"; do
            if [ -d "$op/tests" ]; then
                echo "[WARN] GTest not found; unit tests are skipped (-DBUILD_TESTS=OFF)"
                break
            fi
        done
    fi

    for op in "${OPERATORS[@]}"; do
        if build_single "$op"; then succeeded=$((succeeded + 1)); else failed=$((failed + 1)); fi
    done
    echo "----------------------------------------------------------------"
    echo "Done: $succeeded succeeded, $failed failed."
    [ "$failed" -eq 0 ]
}

main() {
    if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
        usage
        return 0
    fi

    load_repo_config
    discover_operators
    parse_args "$@"

    if [ "$LIST_ONLY" = true ]; then
        list_operators
        return 0
    fi

    if [ -n "$PYTHON_OP" ]; then
        run_python_test "$PYTHON_OP" "$FILELIST"
        return $?
    fi

    if [ "$PYTHON_TESTS" = true ]; then
        run_python_tests_all "$FILELIST" "$PYTHON_TESTS_DEVICE"
        return $?
    fi

    if [ -n "$OPS_BY_NAME" ]; then
        local name resolved
        IFS=',' read -ra name_arr <<< "$OPS_BY_NAME"
        for name in "${name_arr[@]}"; do
            name="${name#"${name%%[![:space:]]*}"}"
            name="${name%"${name##*[![:space:]]}"}"
            if ! resolved="$(resolve_operator "$name")"; then
                log_warn "Operator not found: $name"
                continue
            fi
            OPERATORS+=("$resolved")
        done
    fi

    if [ -n "$FILELIST" ] || [ "$BUILD_ALL" = true ] || [ ${#OPERATORS[@]} -gt 0 ]; then
        run_build
        return $?
    fi

    if [ "$REFERENCE_TEST" = true ]; then
        [ -n "$COMMON_DIRS" ] || COMMON_DIRS="${DEFAULT_COMMON_DIRS[*]:-}"
        # 位置参数是前缀过滤（如 prediction/ProcessControl/），与旧实现一致
        run_reference_tests "$FILELIST" "${OPERATORS[0]:-}" "$COMMON_DIRS"
        return $?
    fi

    echo "build_main.sh: 该子命令尚未迁移到新入口: $*" >&2
    return 2
}

main "$@"
