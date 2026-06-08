#!/usr/bin/env python3
"""
gen_vv_inc.py

Generate .inc files from compiled kernel device_aiv.o files.
Uses the merge_obj_dir/device_aiv.o (merged ELF object, no relocations).

Usage: python3 gen_vv_inc.py
"""

import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

BUILD_DIR = os.path.join(_PROJECT_ROOT, "build")
HOST_DIR = _SCRIPT_DIR

kernels = ["vv_integrate", "vv_finish", "thermo_scale"]

for name in kernels:
    merge_obj = os.path.join(
        BUILD_DIR, "npu_ops", "velocity-verlet",
        f"{name}_kernel_merge_obj_dir", "device_aiv.o"
    )
    inc_path = os.path.join(HOST_DIR, f"{name}_device_aiv_bin.inc")

    if not os.path.exists(merge_obj):
        print(f"[ERROR] {merge_obj} not found!")
        continue

    with open(merge_obj, "rb") as f:
        data = f.read()

    print(f"{name}: merge_obj={len(data)} bytes")

    # Also try the aiv_device_dir for pre-merge version
    aiv_dir = os.path.join(
        BUILD_DIR, "npu_ops", "velocity-verlet",
        f"{name}_kernel_aiv_device_dir", "device_aiv.o"
    )
    if os.path.exists(aiv_dir):
        with open(aiv_dir, "rb") as f2:
            aiv_data = f2.read()
        print(f"  aiv_device_dir version: {len(aiv_data)} bytes")

    # GAFF2 kernel host uses merge_obj version (9392 bytes)
    # We'll use the same approach — embedded raw ELF
    with open(inc_path, "w") as f:
        f.write(f"// Auto-generated from {name}_kernel_merge_obj_dir/device_aiv.o ({len(data)} bytes)\n")
        for i in range(0, len(data)):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{data[i]:02x}")
            if i < len(data) - 1:
                f.write(", ")
            else:
                f.write("\n")
            if i % 12 == 11:
                f.write("\n")

    print(f"  Generated {inc_path}")
    print()

print("Done!")
