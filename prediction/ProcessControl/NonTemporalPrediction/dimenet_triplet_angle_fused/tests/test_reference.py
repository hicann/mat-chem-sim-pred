# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch
from reference.reference import dimenet_triplet_angle_reference


def test_reference_matches_right_angle_triplet():
    position = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    result = dimenet_triplet_angle_reference(
        position,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )
    assert torch.allclose(result, torch.tensor([torch.pi / 2]), atol=1e-6)
