#!/usr/bin/env python3
import numpy as np
import time

def pinn_fc_numpy(inputs, weights_list, bias_list, activation="tanh"):
    act = {"tanh": np.tanh, "sigmoid": lambda x: 1/(1+np.exp(-x)), "relu": lambda x: np.maximum(0, x)}
    f = act[activation]
    h = inputs
    for W, b in zip(weights_list, bias_list):
        h = f(h @ W.T + b)
    return h

def run_benchmark(batch_sizes=None, hidden_dims=None):
    if batch_sizes is None:
        batch_sizes = [16, 64, 256, 1024, 4096]
    if hidden_dims is None:
        hidden_dims = [64, 128, 256]

    print(f"{'Batch':>8} {'Hidden':>8} {'Time(ms)':>10} {'Pts/s':>12}")
    print("-" * 42)

    for B in batch_sizes:
        for H in hidden_dims:
            np.random.seed(0)
            inputs = np.random.randn(B, 3).astype(np.float32)
            W1 = np.random.randn(H, 3).astype(np.float32)
            b1 = np.random.randn(H).astype(np.float32)
            W2 = np.random.randn(H, H).astype(np.float32)
            b2 = np.random.randn(H).astype(np.float32)
            W3 = np.random.randn(1, H).astype(np.float32)
            b3 = np.random.randn(1).astype(np.float32)

            start = time.perf_counter()
            for _ in range(100):
                _ = pinn_fc_numpy(inputs, [W1, W2, W3], [b1, b2, b3], "tanh")
            elapsed = (time.perf_counter() - start) / 100

            print(f"{B:>8} {H:>8} {elapsed*1000:>10.3f} {B/elapsed:>12.0f}")

if __name__ == "__main__":
    run_benchmark()
