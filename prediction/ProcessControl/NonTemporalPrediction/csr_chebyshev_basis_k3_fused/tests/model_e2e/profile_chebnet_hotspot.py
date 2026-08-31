#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Profile the native K=3 Chebyshev basis in a complete Cora ChebNet."""

from __future__ import annotations

import sys
from pathlib import Path

from benchmark_chebnet_cora_e2e import ChebNet, staged_forward
from benchmark_component import native_basis, normalized_csr

sys.path.insert(0, str(Path(__file__).parents[3]))
import spectral_profile_common as profile_common

BASIS_KERNEL_TYPES = {"Index", "InplaceIndexAdd", "Mul", "Pack", "Sub", "ZerosLike"}


def main() -> None:
    args = profile_common.profile_parser().parse_args()
    setup = profile_common.BasisProfileSetup(
        ChebNet,
        16,
        normalized_csr,
        native_basis,
        staged_forward,
        None,
        "two-layer PyG ChebNet, K=3",
        "cora_chebnet_copies",
        BASIS_KERNEL_TYPES,
    )
    profile_common.run_basis_profile(args, setup)


if __name__ == "__main__":
    main()
