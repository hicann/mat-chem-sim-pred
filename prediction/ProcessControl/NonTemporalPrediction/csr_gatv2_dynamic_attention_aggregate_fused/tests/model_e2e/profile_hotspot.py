#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Profile the complete resident-NPU GATv2 model."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


SHARED_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SHARED_ROOT))
COMMON = importlib.import_module("attention_model_e2e_common")
CONFIG = COMMON.ProfileConfig(
    kind="gatv2",
    module_name="benchmark_gatv2_cora_e2e",
    default_copies=(1, 4, 8),
    stage_types=frozenset(
        {"Index", "LeakyRelu", "Mul", "ReduceSum", "SoftmaxV2", "MaskedFill"}
    ),
)


if __name__ == "__main__":
    COMMON.profile_main(Path(__file__).resolve(), CONFIG)
