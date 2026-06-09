#!/usr/bin/env python3
import numpy as np
import time

FNO_PI = 3.14159265358979

def dft_1d(signal):
    n = len(signal)
    re, im = np.zeros(n), np.zeros(n)
    for k in range(n):
        for i in range(n):
            angle = -2.0 * FNO_PI * k * i / n
            re[k] += signal[i] * np.cos(angle)
            im[k] += signal[i] * np.sin(angle)
    return re, im

def idft_1d(re, im):
    n = len(re)
    sig = np.zeros(n)
    for i in range(n):
        for k in range(n):
            angle = 2.0 * FNO_PI * k * i / n
            sig[i] += (re[k] * np.cos(angle) - im[k] * np.sin(angle)) / n
    return sig

def fno_numpy(inputs, lift_w, lift_b, spec_re, spec_im, proj_w, proj_b, grid_dim, modes):
    batch = inputs.shape[0]
    hidden_ch = lift_w.shape[0]
    h = inputs @ lift_w.T + lift_b
    for b in range(batch):
        for c_out in range(hidden_ch):
            h_fft_re = np.zeros(grid_dim)
            h_fft_im = np.zeros(grid_dim)
            for c_in in range(hidden_ch):
                re, im = dft_1d(h[b, c_in])
                m = min(modes, grid_dim // 2 + 1)
                for k in range(m):
                    wr, wi = spec_re[c_out, c_in, k], spec_im[c_out, c_in, k]
                    h_fft_re[k] += wr * re[k] - wi * im[k]
                    h_fft_im[k] += wr * im[k] + wi * re[k]
            h[b, c_out] = idft_1d(h_fft_re, h_fft_im)
    return h @ proj_w.T + proj_b

def run_benchmark():
    configs = [(1, 64), (2, 64), (4, 64), (8, 64), (8, 128)]
    print(f"{'Batch':>6} {'Grid':>6} {'Time(ms)':>10}")
    print("-" * 24)

    for batch, grid in configs:
        in_ch = hidden_ch = out_ch = 4
        modes = grid // 4
        np.random.seed(0)

        inputs = np.random.randn(batch, hidden_ch, grid).astype(np.float32)
        lift_w = np.random.randn(hidden_ch, in_ch).astype(np.float32)
        lift_b = np.random.randn(hidden_ch).astype(np.float32)
        spec_re = np.random.randn(hidden_ch, hidden_ch, modes).astype(np.float32)
        spec_im = np.random.randn(hidden_ch, hidden_ch, modes).astype(np.float32)
        proj_w = np.random.randn(out_ch, hidden_ch).astype(np.float32)
        proj_b = np.random.randn(out_ch).astype(np.float32)

        start = time.perf_counter()
        for _ in range(20):
            _ = fno_numpy(inputs, lift_w, lift_b, spec_re, spec_im,
                           proj_w, proj_b, grid, modes)
        elapsed = (time.perf_counter() - start) / 20
        print(f"{batch:>6} {grid:>6} {elapsed*1000:>10.3f}")

if __name__ == "__main__":
    run_benchmark()
