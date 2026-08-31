# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Reference for PyG-compatible point-pair features."""

from __future__ import annotations

import torch


def _angle(first, second):
    return torch.atan2(
        torch.cross(first, second, dim=1).norm(p=2, dim=1),
        (first * second).sum(dim=1),
    )


def ppf_point_pair_features_reference(position, normal, source_index, target_index):
    source = source_index.long()
    target = target_index.long()
    position_i = position.index_select(0, target)
    position_j = position.index_select(0, source)
    normal_i = normal.index_select(0, target)
    normal_j = normal.index_select(0, source)
    pseudo = position_j - position_i
    return torch.stack(
        [
            pseudo.norm(p=2, dim=1),
            _angle(normal_i, pseudo),
            _angle(normal_j, pseudo),
            _angle(normal_i, normal_j),
        ],
        dim=1,
    )
