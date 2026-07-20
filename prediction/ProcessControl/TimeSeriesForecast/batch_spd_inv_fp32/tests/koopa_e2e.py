#!/usr/bin/env python3
"""End-to-end Koopa forecast: stock (torch.linalg.lstsq -> CPU fallback) vs
custom on-device DMD solve (native bmm + BatchSpdInvFp32 Gram inverse).

Koopa's KPLayer solves the underdetermined system x @ K = y via lstsq; on the
NPU that silently falls back to CPU (1.2-4.3s per call, 3 blocks per forward).
The custom path keeps everything on-device.
"""
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

importlib.import_module("torch_npu")

DEV = "npu:0"
TSLIB = "/root/hd/Time-Series-Library"
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, "/home/ql2025/work/tslib_cann_ops_dev")
sys.path.insert(0, TSLIB)

_KOOPA_VALIDATE_SPEC = importlib.util.spec_from_file_location(
    "koopa_validate_ctypes",
    THIS_DIR / "koopa_validate_ctypes.py",
)
if _KOOPA_VALIDATE_SPEC is None or _KOOPA_VALIDATE_SPEC.loader is None:
    raise ImportError("failed to load koopa_validate_ctypes.py")
_KOOPA_VALIDATE_MODULE = importlib.util.module_from_spec(_KOOPA_VALIDATE_SPEC)
sys.modules["koopa_validate_ctypes"] = _KOOPA_VALIDATE_MODULE
_KOOPA_VALIDATE_SPEC.loader.exec_module(_KOOPA_VALIDATE_MODULE)
spd_inv = _KOOPA_VALIDATE_MODULE.spd_inv


def load_koopa():
    p = f"{TSLIB}/models/Koopa.py"
    spec = importlib.util.spec_from_file_location("KoopaMod", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["KoopaMod"] = mod
    spec.loader.exec_module(mod)
    return mod


def cfg(seq_len, pred_len, enc_in=7):
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=seq_len, label_len=48,
        pred_len=pred_len, enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        features="M", target="OT",
    )


class _Sol:
    def __init__(self, s):
        self.solution = s


def lstsq_custom(x, y, *a, **k):
    # min-norm solution K = x^T (x x^T)^-1 y, Gram inverse via custom op
    xt = x.transpose(-1, -2).contiguous()
    g = torch.matmul(x, xt).contiguous()
    gi = spd_inv(g)
    K = torch.matmul(xt, torch.matmul(gi, y))
    return _Sol(K)


def _ff_forward(self, x):
    # Same semantics as Koopa.FourierFilter, but masks via a real-valued mask
    # (NPU index_put is unsupported for complex64). Applied to BOTH paths so the
    # DMD speedup is isolated.
    xf = torch.fft.rfft(x, dim=1)
    xr = torch.view_as_real(xf)  # [..., 2]
    mask = torch.ones(xf.shape, device=x.device, dtype=xr.dtype)
    mask[:, self.mask_spectrum, :] = 0
    xf2 = torch.view_as_complex((xr * mask.unsqueeze(-1)).contiguous())
    x_var = torch.fft.irfft(xf2, dim=1)
    x_inv = x - x_var
    return x_var, x_inv


def build_model(KoopaMod, c):
    KoopaMod.Model._get_mask_spectrum = lambda self, configs: torch.arange(
        max(1, int((configs.seq_len // 2 + 1) * 0.2)))
    KoopaMod.FourierFilter.forward = _ff_forward
    m = KoopaMod.Model(c).to(DEV).eval()
    return m


def timeit(fn, iters, warmup=1):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def run(KoopaMod, seq_len, pred_len, B, enc_in=7):
    torch.manual_seed(0)
    c = cfg(seq_len, pred_len, enc_in)
    model = build_model(KoopaMod, c)
    x = torch.randn(B, seq_len, enc_in, device=DEV)
    freq = -(-seq_len // pred_len)  # seg_len = pred_len in this model
    m = freq - 1

    orig = torch.linalg.lstsq
    with torch.no_grad():
        # stock (CPU fallback)
        out_stock = model.forecast(x)
        ms_stock = timeit(lambda: model.forecast(x), 3, warmup=1)
        # custom
        torch.linalg.lstsq = lstsq_custom
        try:
            out_custom = model.forecast(x)
            ms_custom = timeit(lambda: model.forecast(x), 20, warmup=2)
        finally:
            torch.linalg.lstsq = orig
    d = (out_stock - out_custom).abs().max().item()
    row = {"seq_len": seq_len, "pred_len": pred_len, "B": B, "enc_in": enc_in,
           "m_snapshots": m, "dynamic_dim": 128,
           "stock_lstsq_cpu_ms": ms_stock, "custom_ms": ms_custom,
           "speedup": ms_stock / ms_custom, "max_diff": d}
    print(f"[e2e] seq{seq_len}/pred{pred_len} B={B} m={m}: stock={ms_stock:.1f}ms "
          f"custom={ms_custom:.2f}ms speedup={row['speedup']:.0f}x diff={d:.2e}", flush=True)
    return row


def main():
    KoopaMod = load_koopa()
    rows = []
    for (seq_len, pred_len, B) in [(336, 96, 32), (720, 96, 32), (512, 96, 16)]:
        rows.append(run(KoopaMod, seq_len, pred_len, B))
    res = {"device": "Ascend 910B3 / node202", "model": "Koopa",
           "note": "stock uses torch.linalg.lstsq (NPU->CPU fallback); custom uses "
                   "native bmm + BatchSpdInvFp32 Gram inverse, all on-device",
           "e2e": rows}
    out = "/home/ql2025/work/tslib_cann_ops_dev/docs/batch_spd_inv_fp32_e2e_koopa.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", out)


if __name__ == "__main__":
    main()
