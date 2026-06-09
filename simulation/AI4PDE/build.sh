#!/bin/bash
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
