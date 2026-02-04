#!/bin/bash
# Build script for LJForceFused operator

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source CANN environment
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

BUILD_TYPE="Release"
if [ "$1" == "debug" ]; then
    BUILD_TYPE="Debug"
fi

if [ "$1" == "clean" ]; then
    rm -rf build
    echo "Cleaned build directory"
    exit 0
fi

mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DSOC_VERSION=Ascend910B1

make -j$(nproc)

echo "Build completed successfully!"
echo "Libraries:"
ls -la *.so lib/*.so 2>/dev/null || true
