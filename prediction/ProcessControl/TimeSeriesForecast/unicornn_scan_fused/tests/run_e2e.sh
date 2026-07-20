#!/bin/bash
set -e
cd /home/ql2025/work/tslib_cann_ops_dev
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CANN=/usr/local/Ascend/ascend-toolkit/latest
M=build/msopgen_unicornn_scan_fused/build_out
SOC=ascend910b
KDIR=$M/op_kernel/kernel
CFG=$KDIR/config/$SOC
if [ ! -f "$CFG/binary_info_config.json" ]; then
  python3 build/msopgen_unicornn_scan_fused/cmake/util/ascendc_ops_config.py -p "$KDIR/$SOC" -s "$SOC" -o "$CFG" 2>/dev/null || true
fi
VENDOR=$M/custom_opp_vendor
rm -rf "$VENDOR"; mkdir -p "$VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"
ln -sfn "$PWD/$KDIR" "$VENDOR/op_impl/ai_core/tbe/kernel"
ln -sfn "$PWD/$M/op_host/libcust_opmaster_rt2.0.so" "$VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
export ASCEND_CUSTOM_OPP_PATH=$PWD/$VENDOR
export LD_LIBRARY_PATH=$PWD/$M/autogen:$PWD/$M/op_api/lib:$CANN/lib64:$CANN/runtime/lib64:$LD_LIBRARY_PATH
echo ===RUN E2E===
python3 cann_ops/unicornn_scan_fused/tests/lem_e2e.py "$@" 2>&1 | tail -30
