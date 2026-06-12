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
import pytest

def pinn_fc_numpy(inputs, weights_list, bias_list, activation="tanh"):
    act = {"tanh": np.tanh, "sigmoid": lambda x: 1/(1+np.exp(-x)), "relu": lambda x: np.maximum(0, x)}
    f = act[activation]
    h = inputs
    for W, b in zip(weights_list, bias_list):
        h = f(h @ W.T + b)
    return h

def pinn_grad_numpy(inputs, weights_list, bias_list, activation="tanh"):
    batch_size, input_dim = inputs.shape
    output_dim = weights_list[-1].shape[0]

    activations = [inputs]
    pre_acts = []
    h = inputs
    for W, b in zip(weights_list, bias_list):
        z = h @ W.T + b
        pre_acts.append(z)
        if activation == "tanh":
            h = np.tanh(z)
        elif activation == "sigmoid":
            h = 1 / (1 + np.exp(-z))
        else:
            h = np.maximum(0, z)
        activations.append(h)

    if activation == "tanh":
        def act_deriv(x): return 1 - np.tanh(x)**2
    elif activation == "sigmoid":
        def act_deriv(x): s = 1/(1+np.exp(-x)); return s*(1-s)
    else:
        def act_deriv(x): return (x > 0).astype(float)

    jacobians = np.zeros((batch_size, output_dim, input_dim))
    for b in range(batch_size):
        for o in range(output_dim):
            grad = np.zeros(output_dim)
            grad[o] = 1.0

            for l in reversed(range(len(weights_list))):
                delta = grad * act_deriv(pre_acts[l][b])
                if l > 0:
                    grad = delta @ weights_list[l]
                else:
                    jacobians[b, o] = delta @ weights_list[l]
    return jacobians

class TestPinnFC:
    def test_forward_small(self):
        np.random.seed(42)
        inputs = np.random.randn(8, 3).astype(np.float32)
        W1 = np.random.randn(16, 3).astype(np.float32)
        b1 = np.random.randn(16).astype(np.float32)
        W2 = np.random.randn(1, 16).astype(np.float32)
        b2 = np.random.randn(1).astype(np.float32)

        expected = pinn_fc_numpy(inputs, [W1, W2], [b1, b2], "tanh")
        assert expected.shape == (8, 1), f"Expected (8,1), got {expected.shape}"
        assert np.all(np.isfinite(expected))

    def test_gradient(self):
        np.random.seed(42)
        inputs = np.random.randn(4, 3).astype(np.float32)
        W1 = np.random.randn(8, 3).astype(np.float32)
        b1 = np.random.randn(8).astype(np.float32)
        W2 = np.random.randn(1, 8).astype(np.float32)
        b2 = np.random.randn(1).astype(np.float32)

        jac = pinn_grad_numpy(inputs, [W1, W2], [b1, b2], "tanh")
        assert jac.shape == (4, 1, 3)
        assert np.all(np.isfinite(jac))

    def test_sigmoid_activation(self):
        np.random.seed(42)
        inputs = np.random.randn(4, 2).astype(np.float32)
        W1 = np.random.randn(8, 2).astype(np.float32)
        b1 = np.random.randn(8).astype(np.float32)
        W2 = np.random.randn(2, 8).astype(np.float32)
        b2 = np.random.randn(2).astype(np.float32)

        out = pinn_fc_numpy(inputs, [W1, W2], [b1, b2], "sigmoid")
        assert out.shape == (4, 2)
        assert np.all((out >= 0) & (out <= 1))

    def test_relu_activation(self):
        np.random.seed(42)
        inputs = np.random.randn(4, 2).astype(np.float32)
        W1 = np.random.randn(8, 2).astype(np.float32)
        b1 = np.random.randn(8).astype(np.float32)
        W2 = np.random.randn(2, 8).astype(np.float32)
        b2 = np.random.randn(2).astype(np.float32)

        out = pinn_fc_numpy(inputs, [W1, W2], [b1, b2], "relu")
        assert out.shape == (4, 2)
        assert np.all(out >= 0)
