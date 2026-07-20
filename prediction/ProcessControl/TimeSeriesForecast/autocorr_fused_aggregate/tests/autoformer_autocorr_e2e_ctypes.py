#!/usr/bin/env python3
import argparse
import ctypes
import importlib
import json
import os
import time
from ctypes import POINTER, byref, c_int, c_int64, c_uint64, c_void_p

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

importlib.import_module("torch_npu")

DEV = "npu:0"
SEQ_LEN = 192
PRED_LEN = 24
INPUT_FEATURES = 64
D_MODEL = 64
HEADS = 4
EMBED = D_MODEL // HEADS
TOP_K = 3
MOVING_AVG = 25
ROOT = "/home/ql2025/work/tslib_cann_ops_dev/build/msopgen_autocorr_fused_aggregate/build_out"
DEFAULT_DATA = "/home/ql2025/work/tslib_cann_ops_dev/data/mamba_forecasting_e2e_mto_4096.csv"
DEFAULT_RESULT = "/home/ql2025/work/tslib_cann_ops_dev/docs/autocorr_autoformer_e2e_results.json"


def init_custom_op():
    torch.empty(1, device=DEV)
    torch.npu.synchronize()
    ctypes.CDLL(f"{ROOT}/autogen/libascend_all_ops.so", mode=ctypes.RTLD_GLOBAL)
    acl = ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    op = ctypes.CDLL(f"{ROOT}/op_api/lib/libcust_opapi.so", mode=ctypes.RTLD_GLOBAL)
    acl.aclCreateTensor.argtypes = [POINTER(c_int64), c_uint64, c_int, POINTER(c_int64), c_int64, c_int, POINTER(c_int64), c_uint64, c_void_p]
    acl.aclCreateTensor.restype = c_void_p
    acl.aclDestroyTensor.argtypes = [c_void_p]
    acl.aclDestroyTensor.restype = c_int
    op.aclnnAutoCorrFusedAggregateGetWorkspaceSize.argtypes = [c_void_p, c_void_p, c_void_p, c_int64, c_void_p, POINTER(c_uint64), POINTER(c_void_p)]
    op.aclnnAutoCorrFusedAggregateGetWorkspaceSize.restype = c_int
    op.aclnnAutoCorrFusedAggregate.argtypes = [c_void_p, c_uint64, c_void_p, c_void_p]
    op.aclnnAutoCorrFusedAggregate.restype = c_int
    return acl, op

_ACL, _OP = init_custom_op()
ACL_FLOAT = 0
ACL_FORMAT_ND = 2


def _acl_tensor(t):
    dims = list(t.shape)
    strides = []
    s = 1
    for d in reversed(dims):
        strides.insert(0, s)
        s *= d
    dims_arr = (c_int64 * len(dims))(*dims)
    strides_arr = (c_int64 * len(strides))(*strides)
    ptr = _ACL.aclCreateTensor(dims_arr, len(dims), ACL_FLOAT, strides_arr, 0, ACL_FORMAT_ND, dims_arr, len(dims), c_void_p(t.data_ptr()))
    if not ptr:
        raise RuntimeError("aclCreateTensor failed")
    return ptr


def autocorr_custom(q, k, v, top_k=TOP_K):
    q, k, v = [x.contiguous() for x in (q, k, v)]
    out = torch.empty_like(v)
    desc = [_acl_tensor(x) for x in (q, k, v, out)]
    workspace_size = c_uint64(0)
    executor = c_void_p()
    ret = _OP.aclnnAutoCorrFusedAggregateGetWorkspaceSize(desc[0], desc[1], desc[2], int(top_k), desc[3], byref(workspace_size), byref(executor))
    if ret:
        raise RuntimeError(f"aclnnAutoCorrFusedAggregateGetWorkspaceSize ret={ret}")
    workspace = torch.empty((int(workspace_size.value),), dtype=torch.uint8, device=q.device) if workspace_size.value else None
    ret = _OP.aclnnAutoCorrFusedAggregate(c_void_p(0 if workspace is None else workspace.data_ptr()), workspace_size.value, executor, c_void_p(torch.npu.current_stream().npu_stream))
    if ret:
        raise RuntimeError(f"aclnnAutoCorrFusedAggregate ret={ret}")
    torch.npu.synchronize()
    for x in desc:
        _ACL.aclDestroyTensor(x)
    return out


def autocorr_framework(q, k, v, top_k=TOP_K):
    length = q.shape[-1]
    scores = []
    for lag in range(length):
        scores.append((q * torch.roll(k, shifts=-lag, dims=-1)).sum(dim=-1))
    score = torch.stack(scores, dim=-1)
    vals, idx = torch.topk(score, k=top_k, dim=-1)
    weights = F.softmax(vals, dim=-1)
    rolled = torch.stack([torch.roll(v, shifts=-lag, dims=-1) for lag in range(length)], dim=-2)
    gather_idx = idx[..., :, None].expand(*idx.shape, length)
    picked = torch.gather(rolled, dim=-2, index=gather_idx)
    return (picked * weights[..., :, None]).sum(dim=-2)


def mark(name, timing, fn):
    if timing is None:
        return fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.npu.synchronize()
    timing[name] = timing.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0
    return out


def make_weights(features, device):
    gen = torch.Generator(device="cpu").manual_seed(20260618)
    weights = {
        "input_proj": torch.randn(D_MODEL, features, generator=gen) * 0.02,
        "q_proj": torch.randn(D_MODEL, D_MODEL, generator=gen) * 0.02,
        "k_proj": torch.randn(D_MODEL, D_MODEL, generator=gen) * 0.02,
        "v_proj": torch.randn(D_MODEL, D_MODEL, generator=gen) * 0.02,
        "out_proj": torch.randn(D_MODEL, D_MODEL, generator=gen) * 0.02,
        "ff1": torch.randn(4 * D_MODEL, D_MODEL, generator=gen) * 0.02,
        "ff1_b": torch.randn(4 * D_MODEL, generator=gen) * 0.01,
        "ff2": torch.randn(D_MODEL, 4 * D_MODEL, generator=gen) * 0.02,
        "ff2_b": torch.randn(D_MODEL, generator=gen) * 0.01,
        "head_w": torch.randn(PRED_LEN * features, D_MODEL, generator=gen) * 0.02,
        "head_b": torch.randn(PRED_LEN * features, generator=gen) * 0.01,
    }
    return {k: v.to(device) for k, v in weights.items()}


def decompose(x):
    # x [B,L,D]
    xp = x.transpose(1, 2)
    pad = MOVING_AVG // 2
    trend = F.avg_pool1d(F.pad(xp, (pad, pad), mode="replicate"), kernel_size=MOVING_AVG, stride=1)
    trend = trend[..., : x.shape[1]].transpose(1, 2).contiguous()
    season = x - trend
    return season, trend


def autoformer_block(x, w, autocorr_impl, timing=None):
    season, trend = mark("series_decomp", timing, lambda: decompose(x))
    q = mark("q_proj", timing, lambda: F.linear(season, w["q_proj"]))
    k = mark("k_proj", timing, lambda: F.linear(season, w["k_proj"]))
    v = mark("v_proj", timing, lambda: F.linear(season, w["v_proj"]))
    def to_bhel(t):
        return t.view(t.shape[0], t.shape[1], HEADS, EMBED).permute(0, 2, 3, 1).contiguous()
    qh, kh, vh = to_bhel(q), to_bhel(k), to_bhel(v)
    ah = mark("autocorr_aggregate", timing, lambda: autocorr_impl(qh, kh, vh, TOP_K))
    a = ah.permute(0, 3, 1, 2).contiguous().view(x.shape[0], x.shape[1], D_MODEL)
    y = mark("out_proj", timing, lambda: F.linear(a, w["out_proj"]))
    h = mark("residual_norm", timing, lambda: F.layer_norm(x + y + trend, (D_MODEL,)))
    ff = mark("ffn", timing, lambda: F.linear(F.gelu(F.linear(h, w["ff1"], w["ff1_b"])), w["ff2"], w["ff2_b"]))
    return mark("ffn_residual_norm", timing, lambda: F.layer_norm(h + ff, (D_MODEL,)))


def forecast_forward(x_raw, w, features, autocorr_impl, timing=None):
    x = mark("input_projection", timing, lambda: F.linear(x_raw, w["input_proj"]))
    h = autoformer_block(x, w, autocorr_impl, timing)
    pred_flat = mark("forecast_head", timing, lambda: F.linear(h[:, -1, :], w["head_w"], w["head_b"]))
    return pred_flat.view(x_raw.shape[0], PRED_LEN, features)


def load_windows(path, features, windows):
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.ffill().bfill().fillna(0.0)
    values = numeric.iloc[:, :features].to_numpy(dtype=np.float32)
    need = SEQ_LEN + PRED_LEN
    if len(values) < need:
        raise ValueError(f"need at least {need} rows, got {len(values)}")
    mean = values[: min(len(values), 2048)].mean(axis=0, keepdims=True)
    std = values[: min(len(values), 2048)].std(axis=0, keepdims=True) + 1e-5
    values = (values - mean) / std
    values = np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    values = np.clip(values, -8.0, 8.0).astype(np.float32)
    max_start = len(values) - need
    starts = np.linspace(0, max_start, num=windows, dtype=np.int64) if windows > 1 else np.array([0], dtype=np.int64)
    xs, ys = [], []
    for s in starts:
        xs.append(torch.from_numpy(values[s : s + SEQ_LEN]).unsqueeze(0).to(DEV))
        ys.append(torch.from_numpy(values[s + SEQ_LEN : s + need]).unsqueeze(0).to(DEV))
    torch.npu.synchronize()
    return {
        "path": path,
        "rows": int(len(values)),
        "all_numeric_columns": int(numeric.shape[1]),
        "feature_count": int(features),
        "windows": int(len(xs)),
        "load_and_preprocess_ms": (time.perf_counter() - t0) * 1000.0,
        "xs": xs,
        "ys": ys,
    }


def evaluate(name, xs, ys, w, features, impl, repeat):
    for x in xs[:2]:
        forecast_forward(x, w, features, impl)
    torch.npu.synchronize()
    loop_samples = []
    last_preds = None
    last_mse = None
    last_mae = None
    for _ in range(repeat):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        preds = []
        mse_vals = []
        mae_vals = []
        for x, y in zip(xs, ys):
            pred = forecast_forward(x, w, features, impl)
            preds.append(pred.detach())
            mse_vals.append(((pred - y) ** 2).mean())
            mae_vals.append((pred - y).abs().mean())
        mse = torch.stack(mse_vals).mean()
        mae = torch.stack(mae_vals).mean()
        torch.npu.synchronize()
        loop_samples.append((time.perf_counter() - t0) * 1000.0)
        last_preds = torch.cat(preds, dim=0)
        last_mse = float(mse.item())
        last_mae = float(mae.item())
    timing = {}
    for x in xs:
        forecast_forward(x, w, features, impl, timing)
    for k in timing:
        timing[k] /= len(xs)
    e2e_ms = sum(loop_samples) / len(loop_samples)
    return {
        "name": name,
        "validation_loop_mean_ms": e2e_ms,
        "validation_loop_min_ms": min(loop_samples),
        "per_window_mean_ms": e2e_ms / len(xs),
        "throughput_windows_per_s": 1000.0 * len(xs) / e2e_ms,
        "mse": last_mse,
        "mae": last_mae,
        "timing_per_window_ms": timing,
        "prediction_sum": float(last_preds.float().sum().item()),
        "predictions": last_preds,
    }


def strip(result):
    out = dict(result)
    out.pop("predictions", None)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--result-json", default=DEFAULT_RESULT)
    parser.add_argument("--features", type=int, default=INPUT_FEATURES)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(0)
    data = load_windows(args.data, args.features, args.windows)
    w = make_weights(args.features, DEV)
    framework = evaluate("framework_naive_autocorr", data["xs"], data["ys"], w, args.features, autocorr_framework, args.repeat)
    custom = evaluate("custom_auto_corr_fused_aggregate", data["xs"], data["ys"], w, args.features, autocorr_custom, args.repeat)
    pred_diff = (framework["predictions"].float() - custom["predictions"].float()).abs()
    ft = framework["timing_per_window_ms"]
    ct = custom["timing_per_window_ms"]
    result = {
        "date": "2026-06-18",
        "device": "Ascend 910B3 / node202",
        "scope": "dataset-level Autoformer-style forecasting E2E: CSV windowing -> AutoCorrelation block -> forecast head -> MSE/MAE",
        "custom_ops_used": ["AutoCorrFusedAggregate"],
        "autocorr_calls_per_forward": 1,
        "dataset": {k: v for k, v in data.items() if k not in ("xs", "ys")},
        "shape": {"B": 1, "seq_len": SEQ_LEN, "pred_len": PRED_LEN, "input_features": args.features, "D_MODEL": D_MODEL, "HEADS": HEADS, "EMBED": EMBED, "TOP_K": TOP_K, "eval_windows": args.windows, "repeat": args.repeat},
        "framework_naive_autocorr": strip(framework),
        "custom_auto_corr_fused_aggregate": strip(custom),
        "comparison": {
            "validation_loop_speedup": framework["validation_loop_mean_ms"] / custom["validation_loop_mean_ms"],
            "per_window_speedup": framework["per_window_mean_ms"] / custom["per_window_mean_ms"],
            "autocorr_speedup": ft["autocorr_aggregate"] / ct["autocorr_aggregate"],
            "mse_abs_diff": abs(framework["mse"] - custom["mse"]),
            "mae_abs_diff": abs(framework["mae"] - custom["mae"]),
            "prediction_max_abs_diff": float(pred_diff.max().item()),
            "prediction_mean_abs_diff": float(pred_diff.mean().item()),
            "framework_autocorr_share_pct": 100.0 * ft["autocorr_aggregate"] / sum(ft.values()),
            "custom_autocorr_share_pct": 100.0 * ct["autocorr_aggregate"] / sum(ct.values()),
        },
    }
    os.makedirs(os.path.dirname(args.result_json), exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
