#!/usr/bin/env python3
import numpy as np
import pytest

def deeponet_numpy(branch_input, trunk_input, branch_weights_list,
                    branch_bias_list, trunk_weights_list, trunk_bias_list):
    def fc_forward(x, weights_list, bias_list):
        h = x
        for W, b in zip(weights_list, bias_list):
            h = np.tanh(h @ W.T + b)
        return h

    branch_code = fc_forward(branch_input, branch_weights_list, branch_bias_list)
    trunk_code = fc_forward(trunk_input, trunk_weights_list, trunk_bias_list)

    result = branch_code @ trunk_code.T
    return result

class TestDeepOnet:
    def test_single_query(self):
        np.random.seed(42)
        batch = 2
        branch_dim = 10
        trunk_dim = 2
        latent_dim = 8
        query = 5

        branch_in = np.random.randn(batch, branch_dim).astype(np.float32)
        trunk_in = np.random.randn(query, trunk_dim).astype(np.float32)

        Wb = np.random.randn(latent_dim, branch_dim).astype(np.float32)
        bb = np.random.randn(latent_dim).astype(np.float32)
        Wt = np.random.randn(latent_dim, trunk_dim).astype(np.float32)
        bt = np.random.randn(latent_dim).astype(np.float32)

        result = deeponet_numpy(branch_in, trunk_in, [Wb], [bb], [Wt], [bt])
        assert result.shape == (batch, query)
        assert np.all(np.isfinite(result))

    def test_single_layer(self):
        np.random.seed(0)
        branch_in = np.random.randn(1, 5).astype(np.float32)
        trunk_in = np.random.randn(3, 2).astype(np.float32)

        Wb = np.random.randn(4, 5).astype(np.float32)
        bb = np.random.randn(4).astype(np.float32)
        Wt = np.random.randn(4, 2).astype(np.float32)
        bt = np.random.randn(4).astype(np.float32)

        result = deeponet_numpy(branch_in, trunk_in, [Wb], [bb], [Wt], [bt])
        assert result.shape == (1, 3)

    def test_same_input_output(self):
        np.random.seed(0)
        branch_in = np.ones((1, 3), dtype=np.float32)
        trunk_in = np.ones((2, 2), dtype=np.float32)

        Wb = np.zeros((4, 3), dtype=np.float32)
        bb = np.zeros(4, dtype=np.float32)
        Wt = np.zeros((4, 2), dtype=np.float32)
        bt = np.zeros(4, dtype=np.float32)

        result = deeponet_numpy(branch_in, trunk_in, [Wb], [bb], [Wt], [bt])
        assert np.allclose(result, 0, atol=1e-6)
