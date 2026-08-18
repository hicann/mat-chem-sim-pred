#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Collect the resident TransformerConv attention hotspot profile."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


COMMON_LOCATION = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMMON_LOCATION))
PROFILE_TOOLS = importlib.import_module("attention_model_e2e_common")


def main() -> None:
    settings = PROFILE_TOOLS.ProfileConfig(
        "transformer",
        "benchmark_transformer_cora_e2e",
        (1, 4, 8),
        frozenset({"Index", "Mul", "ReduceSum", "SoftmaxV2", "MaskedFill"}),
    )
    PROFILE_TOOLS.profile_main(Path(__file__).resolve(), settings)


if __name__ == "__main__":
    main()
