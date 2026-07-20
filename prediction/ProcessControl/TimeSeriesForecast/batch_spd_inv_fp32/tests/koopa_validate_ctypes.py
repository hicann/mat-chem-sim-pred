#!/usr/bin/env python3
"""Correctness + latency validation for BatchSpdInvFp32 and the Koopa DMD path.

  DMD min-norm solve:  K = x^T (x x^T)^-1 y     (x,y: [B,m,E], m<E underdetermined)
  vs torch.linalg.lstsq(x, y).solution (CPU fallback on NPU).

The custom op replaces only the m x m Gram inverse (the part that falls back to
CPU); the two big matmuls run as native torch.bmm on the NPU.
"""
import argparse
import ctypes
import importlib
import json
import time
from ctypes import POINTER, byref, c_int, c_int64, c_uint64, c_void_p

import torch

importlib.import_module("torch_npu")

DEV = "npu:0"
ROOT = "/home/ql2025/work/tslib_cann_ops_dev/build/msopgen_batch_spd_inv_fp32/build_out"


def init_op():
    import glob
    torch.empty(1, device=DEV)
    torch.npu.synchronize()
    for so in glob.glob(f"{ROOT}/**/libascend_all_ops.so", recursive=True):
        ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        break
    ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    acl = ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    op = ctypes.CDLL(f"{ROOT}/op_api/lib/libcust_opapi.so", mode=ctypes.RTLD_GLOBAL)
    acl.aclCreateTensor.argtypes = [POINTER(c_int64), c_uint64, c_int, POINTER(c_int64), c_int64,
                                    c_int, POINTER(c_int64), c_uint64, c_void_p]
    acl.aclCreateTensor.restype = c_void_p
    acl.aclDestroyTensor.argtypes = [c_void_p]
    acl.aclDestroyTensor.restype = c_int
    op.aclnnBatchSpdInvFp32GetWorkspaceSize.argtypes = [c_void_p, c_void_p,
                                                        POINTER(c_uint64), POINTER(c_void_p)]
    op.aclnnBatchSpdInvFp32GetWorkspaceSize.restype = c_int
    op.aclnnBatchSpdInvFp32.argtypes = [c_void_p, c_uint64, c_void_p, c_void_p]
    op.aclnnBatchSpdInvFp32.restype = c_int
    return acl, op


_ACL, _OP = init_op()
ACL_FLOAT = 0
ACL_FORMAT_ND = 2


def _t(t):
    dims = list(t.shape)
    strides = []
    s = 1
    for d in reversed(dims):
        strides.insert(0, s)
        s *= d
    da = (c_int64 * len(dims))(*dims)
    sa = (c_int64 * len(strides))(*strides)
    p = _ACL.aclCreateTensor(da, len(dims), ACL_FLOAT, sa, 0, ACL_FORMAT_ND, da, len(dims), c_void_p(t.data_ptr()))
    if not p:
        raise RuntimeError("aclCreateTensor failed")
    return p


def spd_inv(g):
    # g: [B,m,m] SPD on npu -> gi: [B,m,m]
    g = g.contiguous()
    gi = torch.empty_like(g)
    desc = [_t(g), _t(gi)]
    ws = c_uint64(0)
    ex = c_void_p()
    ret = _OP.aclnnBatchSpdInvFp32GetWorkspaceSize(desc[0], desc[1], byref(ws), byref(ex))
    if ret:
        raise RuntimeError(f"GetWorkspaceSize ret={ret}")
    wbuf = torch.empty((int(ws.value),), dtype=torch.uint8, device=g.device) if ws.value else None
    ret = _OP.aclnnBatchSpdInvFp32(c_void_p(0 if wbuf is None else wbuf.data_ptr()), ws.value, ex,
                                   c_void_p(torch.npu.current_stream().npu_stream))
    if ret:
        raise RuntimeError(f"aclnnBatchSpdInvFp32 ret={ret}")
    torch.npu.synchronize()
    for d in desc:
        _ACL.aclDestroyTensor(d)
    return gi


def dmd_custom(x, y):
    # x,y: [B,m,E]; K = x^T (x x^T)^-1 y, inverse on-device via custom op
    xt = x.transpose(1, 2).contiguous()
    g = torch.bmm(x, xt)
    gi = spd_inv(g)
    return torch.bmm(xt, torch.bmm(gi, y))


def correctness():
    torch.manual_seed(0)
    rows = []
    shapes = [(16, 3, 128), (16, 5, 128), (16, 6, 128), (16, 7, 128), (32, 5, 256), (8, 7, 64)]
    for (B, m, E) in shapes:
        x64 = torch.randn(B, m, E, dtype=torch.float64) * 0.5
        y64 = torch.randn(B, m, E, dtype=torch.float64) * 0.5
        # reference: inverse of Gram + full lstsq min-norm
        g64 = torch.bmm(x64, x64.transpose(1, 2))
        gi64 = torch.linalg.inv(g64)
        K_ref = torch.linalg.lstsq(x64, y64).solution  # min-norm fp64

        x = x64.float().to(DEV)
        y = y64.float().to(DEV)
        g = torch.bmm(x, x.transpose(1, 2).contiguous())
        gi = spd_inv(g)
        d_inv = (gi.cpu().double() - gi64).abs().max().item()
        K = dmd_custom(x, y)
        d_K = (K.cpu().double() - K_ref).abs().max().item()
        # reconstruction residual x@K ~= y
        resid = (torch.bmm(x, K).cpu().double() - y64).abs().max().item()
        rows.append({"B": B, "m": m, "E": E, "max_diff_inv": d_inv,
                     "max_diff_K_vs_lstsq": d_K, "recon_resid": resid})
        print(f"[correct] B={B} m={m} E={E} d_inv={d_inv:.3e} d_K={d_K:.3e} resid={resid:.3e}", flush=True)
    return rows


def bench():
    torch.manual_seed(0)
    rows = []
    for (B, m, E) in [(16, 3, 128), (16, 6, 128), (32, 7, 128), (32, 5, 256)]:
        x = (torch.randn(B, m, E, device=DEV) * 0.5)
        y = (torch.randn(B, m, E, device=DEV) * 0.5)

        def run_custom():
            return dmd_custom(x, y)

        def run_lstsq():
            r = torch.linalg.lstsq(x, y).solution
            torch.npu.synchronize()
            return r

        for _ in range(3):
            run_custom()
        torch.npu.synchronize()

        def timeit(fn, iters):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.npu.synchronize()
            return (time.perf_counter() - t0) * 1000.0 / iters

        ms_c = timeit(run_custom, 50)
        ms_l = timeit(run_lstsq, 5)  # CPU fallback is very slow
        Kc = run_custom()
        Kl = run_lstsq()
        d = (Kc - Kl).abs().max().item()
        row = {"B": B, "m": m, "E": E, "custom_ms": ms_c, "lstsq_cpu_fallback_ms": ms_l,
               "speedup": ms_l / ms_c, "max_diff_vs_lstsq": d}
        rows.append(row)
        print(f"[bench] B={B} m={m} E={E} custom={ms_c:.4f}ms lstsq={ms_l:.1f}ms "
              f"speedup={row['speedup']:.0f}x diff={d:.2e}", flush=True)
    return rows


def robustness():
    # Rank-deficient / collinear Gram must NOT produce NaN/Inf on-device.
    # The pivot-floor guard keeps the output finite & bounded instead of poison
    # propagating through every downstream bmm. (Well-conditioned accuracy is
    # unchanged: see correctness() and the CPU oracle batch_spd_inv_cpu_ref.cpp.)
    torch.manual_seed(0)
    rows = []
    for (B, m, E) in [(16, 5, 80), (16, 6, 96), (8, 3, 48)]:
        x = torch.randn(B, m, E, device=DEV) * 0.5
        x[:, 1, :] = x[:, 0, :]  # row1 = row0 -> rank-deficient Gram
        g = torch.bmm(x, x.transpose(1, 2).contiguous())
        gi = spd_inv(g)
        finite = bool(torch.isfinite(gi).all().item())
        gmax = gi.abs().max().item()
        rows.append({"B": B, "m": m, "E": E, "all_finite": finite, "gi_absmax": gmax})
        print(f"[robust] B={B} m={m} E={E} all_finite={finite} gi_absmax={gmax:.3e}", flush=True)
        assert finite, "rank-deficient Gram produced non-finite gi (pivot-floor guard missing?)"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ql2025/work/tslib_cann_ops_dev/docs/batch_spd_inv_fp32_results.json")
    ap.add_argument("--mode", default="all", choices=["all", "correct", "bench", "robust"])
    args = ap.parse_args()
    res = {"device": "Ascend 910B3 / node202", "op": "BatchSpdInvFp32"}
    if args.mode in ("all", "correct"):
        res["correctness"] = correctness()
    if args.mode in ("all", "robust"):
        res["robustness"] = robustness()
    if args.mode in ("all", "bench"):
        res["bench"] = bench()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
