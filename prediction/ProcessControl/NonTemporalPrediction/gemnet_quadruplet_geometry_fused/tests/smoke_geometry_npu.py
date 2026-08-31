#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Small deterministic geometry binding smoke with explicit angle references."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)

PACKAGE = Path(__file__).parents[1]


def _load_binding():
    path = PACKAGE / "integration" / "geometry_binding.py"
    spec = importlib.util.spec_from_file_location("geometry_binding", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load geometry binding from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class GeometryInputs:
    position: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    interaction_source: torch.Tensor
    interaction_target: torch.Tensor
    reduce_ca: torch.Tensor
    expand_db: torch.Tensor
    reduce_intermediate_ca: torch.Tensor
    expand_intermediate_db: torch.Tensor
    reduce_intermediate_ab: torch.Tensor
    expand_intermediate_ab: torch.Tensor

    def as_tuple(self):
        return tuple(self.__dict__.values())


def _make_inputs():
    position = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    source = torch.tensor([1, 2, 0, 3, 3], dtype=torch.int32)
    target = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)
    small = torch.tensor([0], dtype=torch.int32).repeat(1000)
    return GeometryInputs(
        position.npu(),
        source.npu(),
        target.npu(),
        torch.tensor([1], dtype=torch.int32).npu(),
        torch.tensor([0], dtype=torch.int32).npu(),
        small.npu(),
        torch.tensor([3], dtype=torch.int32).repeat(1000).npu(),
        small.npu(),
        torch.tensor([3], dtype=torch.int32).repeat(1000).npu(),
        small.npu(),
        small.npu(),
    )


def _summarize(values):
    return [
        (
            value.numel(),
            int(torch.count_nonzero(value)),
            float(value.min()),
            float(value.max()),
        )
        for value in values
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    geometry_binding = _load_binding()
    geometry_binding.configure(args.build_dir)
    inputs = _make_inputs()
    output = geometry_binding.gemnet_quadruplet_geometry_fused(*inputs.as_tuple())
    torch.npu.synchronize()
    values = [value.cpu() for value in output]
    LOGGER.info("%s", _summarize(values))
    LOGGER.info("nonzero_indices=%s", torch.nonzero(values[1]).view(-1)[:100].tolist())


if __name__ == "__main__":
    main()
