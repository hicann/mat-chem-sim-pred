#!/bin/bash
set -e
cd /home/ql2025/work/tslib_cann_ops_dev
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CANN=/usr/local/Ascend/ascend-toolkit/latest
M=build/msopgen_sru_scan_fused/build_out
VENDOR=$M/custom_opp_vendor; KDIR=$M/op_kernel/kernel
rm -rf "$VENDOR"; mkdir -p "$VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"
ln -sfn "$PWD/$KDIR" "$VENDOR/op_impl/ai_core/tbe/kernel"
ln -sfn "$PWD/$M/op_host/libcust_opmaster_rt2.0.so" "$VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
export ASCEND_CUSTOM_OPP_PATH=$PWD/$VENDOR
export LD_LIBRARY_PATH=$PWD/$M/autogen:$PWD/$M/op_api/lib:$CANN/lib64:$CANN/runtime/lib64:$LD_LIBRARY_PATH
echo ===COMPILE===
g++ -O2 -std=c++17 -I"$CANN/include" -I"$PWD/$M/op_api/include" \
  cann_ops/sru_scan_fused/tests/sru_scan_fused_probe.cpp \
  -o build/sru_scan_fused_probe \
  -L"$PWD/$M/op_api/lib" -lcust_opapi -L"$CANN/lib64" -lascendcl -lnnopbase -ldl
echo ===RUN PROBE===
./build/sru_scan_fused_probe 4 7 16 16 20 1
./build/sru_scan_fused_probe 8 33 21 32 20 1
./build/sru_scan_fused_probe 16 64 11 64 20 1
./build/sru_scan_fused_probe 8 50 64 64 20 1
