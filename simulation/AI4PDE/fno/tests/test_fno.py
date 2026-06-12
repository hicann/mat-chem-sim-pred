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

FNO_PI = 3.14159265358979

def dft_1d(signal):
    n = len(signal)
    re = np.zeros(n)
    im = np.zeros(n)
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

def fno_numpy(inputs, lift_weight, lift_bias, spectral_weight_real,
              spectral_weight_imag, project_weight, project_bias,
              grid_dim, modes):
    batch_size = inputs.shape[0]
    grid = inputs.shape[1]
    in_ch = inputs.shape[2]
    hidden_ch = lift_weight.shape[0]
    out_ch = project_weight.shape[0]

    h = inputs @ lift_weight.T + lift_bias

    for b in range(batch_size):
        for c_out in range(hidden_ch):
            h_fft_re = np.zeros(grid)
            h_fft_im = np.zeros(grid)
            for c_in in range(hidden_ch):
                re, im = dft_1d(h[b, :, c_in])
                m = min(modes, grid // 2 + 1)
                for k in range(m):
                    wr = spectral_weight_real[c_out, c_in, k]
                    wi = spectral_weight_imag[c_out, c_in, k]
                    h_fft_re[k] += wr * re[k] - wi * im[k]
                    h_fft_im[k] += wr * im[k] + wi * re[k]
            h[b, :, c_out] = idft_1d(h_fft_re, h_fft_im)

    output = h @ project_weight.T + project_bias
    return output

class TestFNO:
    def test_dft_roundtrip(self):
        signal = np.random.randn(8).astype(np.float32)
        re, im = dft_1d(signal)
        recovered = idft_1d(re, im)
        assert np.allclose(signal, recovered, atol=1e-5)

    def test_fno_forward_small(self):
        np.random.seed(42)
        batch = 2
        grid = 8
        in_ch = 2
        hidden_ch = 4
        out_ch = 1
        modes = 4

        inputs = np.random.randn(batch, grid, in_ch).astype(np.float32)
        lift_w = np.random.randn(hidden_ch, in_ch).astype(np.float32)
        lift_b = np.random.randn(hidden_ch).astype(np.float32)
        spec_re = np.random.randn(hidden_ch, hidden_ch, modes).astype(np.float32)
        spec_im = np.random.randn(hidden_ch, hidden_ch, modes).astype(np.float32)
        proj_w = np.random.randn(out_ch, hidden_ch).astype(np.float32)
        proj_b = np.random.randn(out_ch).astype(np.float32)

        result = fno_numpy(inputs, lift_w, lift_b, spec_re, spec_im,
                            proj_w, proj_b, grid, modes)
        assert result.shape == (batch, grid, out_ch)
        assert np.all(np.isfinite(result))

    def test_fno_zero_weights(self):
        np.random.seed(42)
        batch = 1
        grid = 8
        in_ch = 1
        hidden_ch = 2
        out_ch = 1
        modes = 2

        inputs = np.random.randn(batch, grid, in_ch).astype(np.float32)
        lift_w = np.zeros((hidden_ch, in_ch)).astype(np.float32)
        lift_b = np.zeros(hidden_ch).astype(np.float32)
        spec_re = np.zeros((hidden_ch, hidden_ch, modes)).astype(np.float32)
        spec_im = np.zeros((hidden_ch, hidden_ch, modes)).astype(np.float32)
        proj_w = np.zeros((out_ch, hidden_ch)).astype(np.float32)
        proj_b = np.zeros(out_ch).astype(np.float32)

        result = fno_numpy(inputs, lift_w, lift_b, spec_re, spec_im,
                            proj_w, proj_b, grid, modes)
        assert np.allclose(result, 0, atol=1e-6)
