#!/usr/bin/env python3
import numpy as np
import pytest

def mlp_2layer_relu(x, W1, b1, W2, b2):
    h = np.maximum(0, x @ W1.T + b1)
    return h @ W2.T + b2

def mesh_graph_net_numpy(node_features, edge_indices, edge_features,
                          node_W1, node_b1, node_W2, node_b2,
                          node_W3, node_b3, node_W4, node_b4,
                          edge_W1, edge_b1, edge_W2, edge_b2):
    num_nodes = node_features.shape[0]
    node_dim = node_features.shape[1]
    hidden_dim = node_W1.shape[0]
    output_dim = node_W3.shape[0]

    node_update = mlp_2layer_relu(node_features, node_W1, node_b1, node_W2, node_b2)

    edge_agg = np.zeros((num_nodes, hidden_dim))
    neighbor_counts = np.zeros(num_nodes, dtype=int)

    for e in range(edge_indices.shape[0]):
        src, dst = edge_indices[e]
        concat = np.concatenate([node_features[src], edge_features[e]])
        msg = mlp_2layer_relu(concat, edge_W1, edge_b1, edge_W2, edge_b2)
        edge_agg[dst] += msg
        neighbor_counts[dst] += 1

    for i in range(num_nodes):
        if neighbor_counts[i] > 0:
            edge_agg[i] /= neighbor_counts[i]

    h = node_update + edge_agg
    output = mlp_2layer_relu(h, node_W3, node_b3, node_W4, node_b4)
    return output

class TestMeshGraphNet:
    def test_small_graph(self):
        np.random.seed(42)
        num_nodes = 5
        node_dim = 4
        edge_dim = 2
        hidden_dim = 8
        output_dim = 2

        edge_indices = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4],
                                  [1, 0], [2, 1], [3, 2], [4, 3], [4, 0]],
                                 dtype=np.int32)

        node_feat = np.random.randn(num_nodes, node_dim).astype(np.float32)
        edge_feat = np.random.randn(edge_indices.shape[0], edge_dim).astype(np.float32)

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

        result = mesh_graph_net_numpy(
            node_feat, edge_indices, edge_feat,
            nW1, nb1, nW2, nb2,
            nW3, nb3, nW4, nb4,
            eW1, eb1, eW2, eb2)

        assert result.shape == (num_nodes, output_dim)
        assert np.all(np.isfinite(result))

    def test_no_edges(self):
        np.random.seed(0)
        node_feat = np.random.randn(3, 2).astype(np.float32)
        edge_indices = np.zeros((0, 2), dtype=np.int32)
        edge_feat = np.zeros((0, 1), dtype=np.float32)

        hidden_dim = 4
        nW1 = np.random.randn(hidden_dim, 2).astype(np.float32)
        nb1 = np.zeros(hidden_dim, dtype=np.float32)
        nW2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        nb2 = np.zeros(hidden_dim, dtype=np.float32)
        nW3 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        nb3 = np.zeros(hidden_dim, dtype=np.float32)
        nW4 = np.random.randn(2, hidden_dim).astype(np.float32)
        nb4 = np.zeros(2, dtype=np.float32)
        eW1 = np.random.randn(hidden_dim, 3).astype(np.float32)
        eb1 = np.zeros(hidden_dim, dtype=np.float32)
        eW2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        eb2 = np.zeros(hidden_dim, dtype=np.float32)

        result = mesh_graph_net_numpy(
            node_feat, edge_indices, edge_feat,
            nW1, nb1, nW2, nb2,
            nW3, nb3, nW4, nb4,
            eW1, eb1, eW2, eb2)

        assert result.shape == (3, 2)
