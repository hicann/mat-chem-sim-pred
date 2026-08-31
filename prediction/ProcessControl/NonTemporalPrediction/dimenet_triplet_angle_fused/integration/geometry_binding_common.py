# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Shared ctypes loading helpers for molecular geometry operators."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def load_host_library(build_dir, host_name, kernel_pattern):
    build = Path(build_dir)
    if (build / "build").is_dir():
        build = build / "build"
    toolkit = Path(
        os.environ.get(
            "ASCEND_CANN_PACKAGE_PATH",
            os.environ.get(
                "ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest"
            ),
        )
    )
    ascendcl = toolkit / "lib64" / "libascendcl.so"
    if ascendcl.exists():
        ctypes.CDLL(str(ascendcl), mode=ctypes.RTLD_GLOBAL)
    for directory in (build / "lib", build):
        for library in directory.glob(kernel_pattern):
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    candidates = (build / host_name, build / "lib" / host_name)
    for candidate in candidates:
        if candidate.exists():
            return ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
    raise FileNotFoundError(f"cannot find {host_name} under {build}")
