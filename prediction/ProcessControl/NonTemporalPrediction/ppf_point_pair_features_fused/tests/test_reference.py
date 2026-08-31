# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import torch

from reference.reference import ppf_point_pair_features_reference


def test_reference_matches_pyg_definition():
    position = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normal = torch.nn.functional.normalize(position + 0.25, dim=-1)
    source = torch.tensor([1, 2], dtype=torch.int32)
    target = torch.tensor([0, 0], dtype=torch.int32)
    output = ppf_point_pair_features_reference(position, normal, source, target)
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()
