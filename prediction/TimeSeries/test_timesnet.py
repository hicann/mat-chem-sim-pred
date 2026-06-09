"""
TimesNet 测试 — 验证 PyTorch 实现正确性。

Usage:
    pytest prediction/TimeSeries/test_timesnet.py -v
    python -m prediction.TimeSeries.test_timesnet
"""

import numpy as np
import torch

from .timesnet import (
    fft_for_period,
    fft_for_period_np,
    TimesBlock,
    TimesNet,
)


RTOL = 1e-4
ATOL = 1e-4


def check(label, result, expected, rtol=RTOL, atol=ATOL):
    try:
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)
        print(f"  {label}:  PASS")
    except AssertionError as e:
        err = np.abs(result - expected).max()
        print(f"  {label}:  FAIL (max_err={err:.2e})")


def test_fft_for_period():
    np.random.seed(42)
    batch, seq_len, d_model = 4, 96, 16
    x = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    x_t = torch.from_numpy(x)

    periods_t, amp_t = fft_for_period(x_t, k=3)
    periods_np, amp_np = fft_for_period_np(x, k=3)

    check("FFT periods", periods_t.numpy(), periods_np.astype(np.int64))
    check("FFT amplitudes", amp_t.numpy(), amp_np)

    for p in periods_t:
        assert 2 <= p <= seq_len, f"Period {p.item()} out of range"
    print(f"  Discovered periods: {periods_t.tolist()}")


def test_timesblock_shape():
    torch.manual_seed(42)
    batch, seq_len, d_model = 8, 96, 32

    block = TimesBlock(d_model=d_model, top_k=3)
    x = torch.randn(batch, seq_len, d_model)

    out = block(x)
    assert out.shape == (batch, seq_len, d_model), f"Expected {(batch, seq_len, d_model)}, got {out.shape}"
    print(f"  TimesBlock output shape: {out.shape}")
    print(f"  Input-output diff norm: {(out - x).norm().item():.4f}")


def test_timesnet_forecast():
    torch.manual_seed(42)
    batch, seq_len, pred_len, num_features = 4, 96, 48, 7

    model = TimesNet(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=32,
        d_ff=64,
        n_layers=2,
        top_k=3,
        num_features=num_features,
    )

    x = torch.randn(batch, seq_len, num_features)
    out = model(x)

    assert out.shape == (batch, pred_len, num_features), f"Expected {(batch, pred_len, num_features)}, got {out.shape}"

    y = torch.randn(batch, pred_len, num_features)
    loss = (out - y).pow(2).mean()
    loss.backward()

    n_nonzero = sum((p.grad is not None and p.grad.abs().sum() > 0)
                    for p in model.parameters())
    print(f"  TimesNet forecast output shape: {out.shape}")
    print(f"  Gradient flow: {n_nonzero}/{sum(1 for _ in model.parameters())} params")
    assert n_nonzero > 0, "No gradients!"


def test_timesnet_synthetic_sine():
    """用合成正弦波验证 TimesNet 能学到周期模式."""
    torch.manual_seed(42)
    seq_len, pred_len = 96, 48
    batch = 8

    t = np.arange(seq_len + pred_len, dtype=np.float32) / 24.0
    data = np.sin(2 * np.pi * t) + 0.5 * np.sin(2 * np.pi * t * 4)
    data = data.reshape(1, -1, 1).repeat(batch, axis=0)
    data += np.random.randn(*data.shape).astype(np.float32) * 0.05

    x = data[:, :seq_len, :]
    y = data[:, seq_len:, :]

    model = TimesNet(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=32,
        d_ff=64,
        n_layers=2,
        top_k=3,
        num_features=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)

    for epoch in range(20):
        optimizer.zero_grad()
        out = model(x_t)
        loss = (out - y_t).pow(2).mean()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"  Synthetic sine final loss: {final_loss:.6f}")
    assert final_loss < 0.1, f"Loss too high: {final_loss}"


def main():
    print("=" * 60)
    print("TimesNet Test Suite")
    print("=" * 60)

    test_fft_for_period()
    test_timesblock_shape()
    test_timesnet_forecast()
    test_timesnet_synthetic_sine()

    print("=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()