#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

BUILD_TYPE="Release"
TARGET="all"
if [ "$1" == "debug" ]; then
    BUILD_TYPE="Debug"
elif [ "$1" == "clean" ]; then
    rm -rf build
    echo "Cleaned build directory"
    exit 0
elif [ "$1" == "pinn" ]; then
    TARGET="pinn"
elif [ "$1" == "fno" ]; then
    TARGET="fno"
elif [ "$1" == "deeponet" ]; then
    TARGET="deeponet"
elif [ "$1" == "mesh" ]; then
    TARGET="mesh_graph_net"
fi

mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DSOC_VERSION=Ascend910B1 \
    -DBUILD_TARGET=$TARGET

make -j$(nproc)

echo "Build completed successfully!"
echo "Libraries:"
find . -name "*.so" -exec ls -la {} \;
