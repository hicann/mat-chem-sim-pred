#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import time

def mlp_2layer_relu(x, W1, b1, W2, b2):
    h = np.maximum(0, x @ W1.T + b1)
    return h @ W2.T + b2

def mesh_graph_net_numpy(node_feat, edge_idx, edge_feat,
                          nW1, nb1, nW2, nb2,
                          nW3, nb3, nW4, nb4,
                          eW1, eb1, eW2, eb2):
    num_nodes = node_feat.shape[0]
    hidden_dim = nW1.shape[0]
    node_update = mlp_2layer_relu(node_feat, nW1, nb1, nW2, nb2)
    edge_agg = np.zeros((num_nodes, hidden_dim))
    counts = np.zeros(num_nodes, dtype=int)
    for e in range(edge_idx.shape[0]):
        src, dst = edge_idx[e]
        concat = np.concatenate([node_feat[src], edge_feat[e]])
        msg = mlp_2layer_relu(concat, eW1, eb1, eW2, eb2)
        edge_agg[dst] += msg
        counts[dst] += 1
    for i in range(num_nodes):
        if counts[i] > 0:
            edge_agg[i] /= counts[i]
    h = node_update + edge_agg
    return mlp_2layer_relu(h, nW3, nb3, nW4, nb4)

def run_benchmark():
    configs = [(100, 400), (500, 2000), (1000, 5000), (2000, 10000)]
    print(f"{'Nodes':>6} {'Edges':>8} {'Time(ms)':>10}")
    print("-" * 26)
    for n, m in configs:
        node_dim, edge_dim, hidden_dim, output_dim = 8, 4, 32, 4
        np.random.seed(0)
        node_feat = np.random.randn(n, node_dim).astype(np.float32)
        edge_idx = np.random.randint(0, n, (m, 2)).astype(np.int32)
        edge_feat = np.random.randn(m, edge_dim).astype(np.float32)
        nW1 = np.random.randn(hidden_dim, node_dim).astype(np.float32)
        nb1 = np.zeros(hidden_dim, dtype=np.float32)
        nW2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        nb2 = np.zeros(hidden_dim, dtype=np.float32)
        nW3 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        nb3 = np.zeros(hidden_dim, dtype=np.float32)
        nW4 = np.random.randn(output_dim, hidden_dim).astype(np.float32)
        nb4 = np.zeros(output_dim, dtype=np.float32)
        eW1 = np.random.randn(hidden_dim, node_dim + edge_dim).astype(np.float32)
        eb1 = np.zeros(hidden_dim, dtype=np.float32)
        eW2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        eb2 = np.zeros(hidden_dim, dtype=np.float32)
        start = time.perf_counter()
        for _ in range(20):
            _ = mesh_graph_net_numpy(
                node_feat, edge_idx, edge_feat,
                nW1, nb1, nW2, nb2,
                nW3, nb3, nW4, nb4,
                eW1, eb1, eW2, eb2)
        elapsed = (time.perf_counter() - start) / 20
        print(f"{n:>6} {m:>8} {elapsed*1000:>10.3f}")

if __name__ == "__main__":
    run_benchmark()
