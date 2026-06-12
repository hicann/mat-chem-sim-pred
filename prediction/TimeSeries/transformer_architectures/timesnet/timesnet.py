"""
TimesNet: Temporal 2D-Variation Modeling — PyTorch Reference Implementation.

Port of https://github.com/thuml/TimesNet
Core idea: use FFT to discover periods, reshape 1D sequences to 2D tensors,
then apply 2D convolutions to capture intra-period and inter-period variations.

Reference: Wu et al., ICLR 2023 (https://arxiv.org/abs/2210.02186)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# FFT-based period discovery
# ============================================================================

def fft_for_period_np(x: np.ndarray, k: int = 3) -> tuple:
    """NumPy reference: discover top-k periods via FFT.

    Args:
        x: (batch, seq_len, d_model)
        k: number of top periods to select

    Returns:
        periods: (k,) — discovered periods
        amplitudes: (batch, k) — amplitude for each period
    """
    xf = np.fft.rfft(x, axis=1)
    frequency_amp = np.abs(xf).mean(0).mean(-1)
    frequency_amp[0] = 0
    top_k_idx = np.argsort(frequency_amp)[-k:][::-1]
    periods = x.shape[1] // top_k_idx
    amplitudes = np.abs(xf).mean(-1)[:, top_k_idx]
    return periods, amplitudes


def fft_for_period(x: torch.Tensor, k: int = 3) -> tuple:
    """Discover top-k periods via FFT (PyTorch).

    Returns:
        periods: (k,) — periods (lengths)
        amplitudes: (batch, k)
    """
    xf = torch.fft.rfft(x, dim=1)
    frequency_amp = xf.abs().mean(0).mean(-1)
    frequency_amp[0] = 0
    top_k_idx = torch.topk(frequency_amp, k).indices
    periods = x.shape[1] // top_k_idx
    amplitudes = xf.abs().mean(-1)[:, top_k_idx]
    return periods, amplitudes


# ============================================================================
# Inception Block — 核心 2D 卷积模块
# ============================================================================

class InceptionBlock(nn.Module):
    """Multi-scale 2D convolution block inspired by Inception."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_sizes: tuple = (3, 5, 7), padding: str = "same"):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, k, padding=padding)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Conv2d(len(kernel_sizes) * out_channels + in_channels,
                              out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_outs = [conv(x) for conv in self.convs]
        pool_out = self.maxpool(x)
        cat = torch.cat([*conv_outs, pool_out], dim=1)
        return self.proj(cat)


# ============================================================================
# TimesBlock — 单个 TimesNet 块
# ============================================================================

class TimesBlock(nn.Module):
    """
    TimesBlock: 1D → (reshape) → 2D Conv → (reshape) → 1D.

    For each discovered period, the sequence is reshaped into a 2D tensor,
    processed by an Inception block, then reshaped back.
    """

    def __init__(self, d_model: int, top_k: int = 3, conv_kernel_sizes: tuple = (3, 5, 7)):
        super().__init__()
        self.top_k = top_k
        self.conv = InceptionBlock(d_model, d_model, conv_kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            out: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        periods, amplitudes = fft_for_period(x, self.top_k)

        out_res = []
        for i in range(self.top_k):
            period = periods[i].item()
            if period <= 1 or period >= seq_len:
                out_res.append(x)
                continue

            n_periods = seq_len // period
            n_remain = seq_len - n_periods * period

            # reshape: (batch, d_model, n_periods, period)
            x_2d = x[:, :n_periods * period, :]
            x_2d = x_2d.permute(0, 2, 1)
            x_2d = x_2d.reshape(batch, d_model, n_periods, period)

            # 2D convolution
            x_2d = self.conv(x_2d)

            # reshape back: (batch, n_periods * period, d_model)
            x_1d = x_2d.reshape(batch, d_model, n_periods * period).permute(0, 2, 1)

            if n_remain > 0:
                x_1d = torch.cat([x_1d, x[:, -n_remain:, :]], dim=1)

            out_res.append(x_1d)

        # adaptive aggregation by amplitude
        amplitudes = F.softmax(amplitudes, dim=-1)
        out = torch.zeros_like(x)
        for i in range(self.top_k):
            out = out + out_res[i] * amplitudes[:, i:i+1, None]

        return out + x  # residual


# ============================================================================
# TimesNet Model
# ============================================================================

class TimesNet(nn.Module):
    """
    TimesNet for time series forecasting.

    Args:
        seq_len: Input sequence length.
        pred_len: Prediction horizon.
        d_model: Hidden dimension.
        d_ff: Feed-forward dimension.
        n_layers: Number of TimesBlocks.
        top_k: Number of periods to extract.
        num_features: Number of input features (channels).
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        d_model: int = 64,
        d_ff: int = 128,
        n_layers: int = 2,
        top_k: int = 3,
        num_features: int = 7,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.embedding = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(d_model, top_k) for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, pred_len * num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_features)

        Returns:
            out: (batch, pred_len, num_features)
        """
        x = self.embedding(x)

        for block, norm in zip(self.blocks, self.layer_norms):
            x = norm(block(x))

        x = x[:, -1, :]
        out = self.predictor(x)
        out = out.reshape(-1, self.pred_len, x.shape[-1])
        return out


# ============================================================================
# NumPy reference: TimesNet 单步推理
# ============================================================================

def timesnet_block_np(x: np.ndarray, W_conv: dict, top_k: int = 3) -> np.ndarray:
    """NumPy reference for TimesBlock forward (simplified, single period)."""
    batch, seq_len, d_model = x.shape

    xf = np.fft.rfft(x, axis=1)
    frequency_amp = np.abs(xf).mean(0).mean(-1)
    frequency_amp[0] = 0
    top_k_idx = np.argsort(frequency_amp)[-top_k:][::-1]
    period = seq_len // top_k_idx[0]
    n_periods = seq_len // period
    x_trimmed = x[:, :n_periods * period, :]

    x_2d = x_trimmed.transpose(0, 2, 1).reshape(batch, d_model, n_periods, period)

    conv_out = W_conv["w"] @ x_2d + W_conv["b"][:, None, None]
    conv_out = np.maximum(conv_out, 0)

    x_1d = conv_out.reshape(batch, d_model, n_periods * period).transpose(0, 2, 1)
    return x_1d + x[:, :n_periods * period, :]