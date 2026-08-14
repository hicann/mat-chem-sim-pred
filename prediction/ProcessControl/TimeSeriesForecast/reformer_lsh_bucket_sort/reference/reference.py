# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Dependency-free CPU reference for ReformerLshBucketSort."""


def _matrix(value, name):
    if not isinstance(value, (list, tuple)) or not value or not value[0]:
        raise ValueError(f"{name} must be a non-empty matrix")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError(f"{name} must be rectangular")
    return len(value), width


def reference(keys, sequence_length, total_buckets):
    _matrix(keys, "keys")
    if sequence_length <= 0 or not 1 <= total_buckets <= 4096:
        raise ValueError("invalid bucket attributes")
    sorted_keys, stickers, inverses = [], [], []
    for row in keys:
        if any(key < 0 or key // sequence_length >= total_buckets for key in row):
            raise ValueError("key outside encoded bucket range")
        sticker = sorted(range(len(row)), key=lambda i: row[i] // sequence_length)
        inverse = [0] * len(row)
        for sorted_position, source_position in enumerate(sticker):
            inverse[source_position] = sorted_position
        sorted_keys.append([row[i] for i in sticker])
        stickers.append(sticker)
        inverses.append(inverse)
    return sorted_keys, stickers, inverses
