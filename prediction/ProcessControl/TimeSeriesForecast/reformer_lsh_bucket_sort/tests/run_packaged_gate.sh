#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -eo pipefail

OP_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK_ROOT=${1:?usage: run_packaged_gate.sh <new-work-dir> [physical-device]}
DEVICE=${2:-0}
if [ -e "$WORK_ROOT" ]; then
    echo "work directory already exists: $WORK_ROOT" >&2
    exit 17
fi
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
fi
set -u

mkdir -p "$WORK_ROOT/common"
chmod go-w "$WORK_ROOT" "$WORK_ROOT/common"
cp "$OP_ROOT/msopgen/reformer_lsh_bucket_sort_msopgen.json" "$WORK_ROOT/schema.json"
chmod go-w "$WORK_ROOT/schema.json"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
    -s "$OP_ROOT/tests" -p 'test_*.py' -v
msopgen gen -i "$WORK_ROOT/schema.json" -f aclnn -c ai_core-ascend910b \
    -out "$WORK_ROOT/project" -lan cpp

PROJECT="$WORK_ROOT/project"
mv "$PROJECT/op_host/reformer_lsh_bucket_sort.cpp" \
    "$PROJECT/op_host/reformer_lsh_bucket_sort.generated.cpp.stub"
cp "$OP_ROOT/op_host/reformer_lsh_bucket_sort_host.cpp" "$PROJECT/op_host/"
cp "$OP_ROOT/op_host/reformer_lsh_bucket_sort_def.cpp" "$PROJECT/op_host/"
cp "$OP_ROOT/op_host/reformer_lsh_bucket_sort_tiling.h" "$PROJECT/op_host/"
cp "$OP_ROOT/op_kernel/reformer_lsh_bucket_sort_kernel.cpp" \
    "$PROJECT/op_kernel/reformer_lsh_bucket_sort.cpp"
cp "$OP_ROOT/../common/op_def_utils.h" "$WORK_ROOT/common/"
cp "$OP_ROOT/../common/op_host_utils.h" "$WORK_ROOT/common/"

cmake -S "$PROJECT" -B "$PROJECT/build_formal" \
    -DASCEND_COMPUTE_UNIT=ascend910b -DCMAKE_BUILD_TYPE=Release
cmake --build "$PROJECT/build_formal" --target install -- -j2

g++ -std=c++17 -O2 "$OP_ROOT/examples/test_aclnn_reformer_lsh_bucket_sort.cpp" \
    -I"$PROJECT/build_out/op_api/include" \
    -I/usr/local/Ascend/ascend-toolkit/latest/include \
    -L"$PROJECT/build_out/op_api/lib" \
    -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
    -Wl,-rpath,"$PROJECT/build_out/op_api/lib" \
    -lcust_opapi -lnnopbase -lascendcl -ldl \
    -o "$WORK_ROOT/test_aclnn_reformer_lsh_bucket_sort"

ASCEND_RT_VISIBLE_DEVICES="$DEVICE" \
LD_LIBRARY_PATH="$PROJECT/build_formal/autogen:$PROJECT/build_out/op_api/lib:${LD_LIBRARY_PATH:-}" \
    "$WORK_ROOT/test_aclnn_reformer_lsh_bucket_sort"
