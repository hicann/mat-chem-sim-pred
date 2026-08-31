# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
from reference.reference import gemnet_geometry_reference


def test_reference_returns_official_cache_shapes_and_finite_values():
    position = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    source = torch.tensor([1, 2, 0, 3], dtype=torch.int32)
    target = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    interaction_source = torch.tensor([1], dtype=torch.int32)
    interaction_target = torch.tensor([0], dtype=torch.int32)
    one = torch.tensor([0], dtype=torch.int32)
    values = gemnet_geometry_reference(
        position,
        source,
        target,
        interaction_source,
        interaction_target,
        one,
        one,
        one,
        torch.tensor([3], dtype=torch.int32),
        one,
        one,
    )
    assert [value.shape for value in values] == [torch.Size([1])] * 3
    assert all(torch.isfinite(value).all() for value in values)
