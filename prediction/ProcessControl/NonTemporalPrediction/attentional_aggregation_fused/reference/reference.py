# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def attentional_aggregation(graph_ptr, features, gates):
    output = np.empty((graph_ptr.size - 1, features.shape[1]), dtype=np.float32)
    for graph in range(graph_ptr.size - 1):
        begin, end = int(graph_ptr[graph]), int(graph_ptr[graph + 1])
        logits = gates[begin:end]
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        output[graph] = np.sum(weights[:, None] * features[begin:end], axis=0)
    return output


def use_custom_path(nodes, graphs, channels, dtype, gates_are_finite=True):
    return (
        nodes > 0
        and graphs >= 4
        and 0 < channels <= 1024
        and np.dtype(dtype) == np.dtype(np.float32)
        and gates_are_finite
    )
