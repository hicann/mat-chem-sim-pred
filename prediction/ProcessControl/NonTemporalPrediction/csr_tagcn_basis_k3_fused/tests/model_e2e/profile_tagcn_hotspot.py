#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Profile the native three-hop basis in the complete Cora TAGCN."""

from __future__ import annotations

import sys
from pathlib import Path

import torch_geometric.transforms as T
from benchmark_component import native_basis, normalized_csr
from benchmark_tagcn_cora_e2e import TagNet, staged_forward

sys.path.insert(0, str(Path(__file__).parents[3]))
import spectral_profile_common as profile_common

BASIS_KERNEL_TYPES = {"Index", "InplaceIndexAdd", "Mul", "Pack", "ZerosLike"}


def main() -> None:
    args = profile_common.profile_parser().parse_args()
    setup = profile_common.BasisProfileSetup(
        model_type=TagNet,
        hidden=16,
        layout=normalized_csr,
        native=native_basis,
        staged_forward=staged_forward,
        transform=T.NormalizeFeatures(),
        model_label="maintained PyG examples/tagcn.py two-layer TAGCN, K=3",
        trace_prefix="cora_tagcn_copies",
        kernel_types=BASIS_KERNEL_TYPES,
    )
    profile_common.run_basis_profile(args, setup)


if __name__ == "__main__":
    main()
