#!/usr/bin/env python3
"""CfC encoder E2E: replace the per-timestep recurrence loop with CfcScanFused.

Builds a faithful CfC (Closed-form Continuous-time, Hasani et al. 2022) closed-form
recurrence as the framework baseline on NPU, then replaces each layer's sequential
time loop with the fused custom op. Reports:
  (1) correctness: fused encoder output vs fp32 torch reference
  (2) component speedup: torch single-layer time loop vs fused op (one layer)
  (3) E2E speedup: torch full N-layer encoder vs fused full N-layer encoder

Weight convention (matches kernel + probe): Wcat is the transposed, column-packed
projection [H+IN, 3H]:  cols[0:H]=Wf1, cols[H:2H]=Wf2, cols[2H:3H]=Wt(gate),
acting on z=[h;x_s]. bcat=[bf1|bf2|bt].
"""
import ctypes
import glob
import importlib
import json
import sys
import time
from ctypes import POINTER, byref, c_int, c_int64, c_uint64, c_void_p

import numpy as np
import torch

importlib.import_module("torch_npu")

DEV = "npu:0"
ROOT = "/home/ql2025/work/tslib_cann_ops_dev/build/msopgen_cfc_scan_fused/build_out"
ACL_FLOAT, ACL_FORMAT_ND = 0, 2


def init_op():
    torch.empty(1, device=DEV); torch.npu.synchronize()
    for so in glob.glob(f"{ROOT}/**/libascend_all_ops.so", recursive=True):
        ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL); break
    ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    op = ctypes.CDLL(f"{ROOT}/op_api/lib/libcust_opapi.so", mode=ctypes.RTLD_GLOBAL)
    acl = ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    acl.aclCreateTensor.argtypes = [POINTER(c_int64), c_uint64, c_int, POINTER(c_int64), c_int64,
                                    c_int, POINTER(c_int64), c_uint64, c_void_p]
    acl.aclCreateTensor.restype = c_void_p
    acl.aclDestroyTensor.argtypes = [c_void_p]; acl.aclDestroyTensor.restype = c_int
    op.aclnnCfcScanFusedGetWorkspaceSize.argtypes = [c_void_p]*4 + [POINTER(c_uint64), POINTER(c_void_p)]
    op.aclnnCfcScanFusedGetWorkspaceSize.restype = c_int
    op.aclnnCfcScanFused.argtypes = [c_void_p, c_uint64, c_void_p, c_void_p]
    op.aclnnCfcScanFused.restype = c_int
    return acl, op


_ACL, _OP = init_op()


def _t(t):
    dims = list(t.shape); strides = []; s = 1
    for d in reversed(dims):
        strides.insert(0, s); s *= d
    da = (c_int64 * len(dims))(*dims); sa = (c_int64 * len(strides))(*strides)
    p = _ACL.aclCreateTensor(da, len(dims), ACL_FLOAT, sa, 0, ACL_FORMAT_ND, da, len(dims), c_void_p(t.data_ptr()))
    if not p: raise RuntimeError("aclCreateTensor failed")
    return p


def custom_cfc(x, Wcat, bcat):
    """x[B,L,IN] Wcat[H+IN,3H] bcat[3H] -> out[B,L,H]."""
    x = x.contiguous(); Wcat = Wcat.contiguous(); bcat = bcat.contiguous()
    B, L, IN = x.shape; H = Wcat.shape[1] // 3
    out = torch.zeros(B, L, H, device=DEV)
    desc = [_t(x), _t(Wcat), _t(bcat), _t(out)]
    ws = c_uint64(0); ex = c_void_p()
    r = _OP.aclnnCfcScanFusedGetWorkspaceSize(desc[0], desc[1], desc[2], desc[3], byref(ws), byref(ex))
    if r: raise RuntimeError(f"GetWorkspaceSize ret={r}")
    wbuf = torch.empty((int(ws.value),), dtype=torch.uint8, device=DEV) if ws.value else None
    r = _OP.aclnnCfcScanFused(c_void_p(0 if wbuf is None else wbuf.data_ptr()), ws.value, ex,
                              c_void_p(torch.npu.current_stream().npu_stream))
    if r: raise RuntimeError(f"aclnnCfcScanFused ret={r}")
    torch.npu.synchronize()
    for d in desc: _ACL.aclDestroyTensor(d)
    return out


def torch_cfc_loop(x, Wcat, bcat, H):
    """Faithful CfC closed-form per-timestep loop (framework baseline). x[B,L,IN] -> out[B,L,H]."""
    B, L, IN = x.shape
    h = torch.zeros(B, H, device=DEV)
    Wf1 = Wcat[:, 0:H]; Wf2 = Wcat[:, H:2 * H]; Wt = Wcat[:, 2 * H:3 * H]
    bf1 = bcat[0:H]; bf2 = bcat[H:2 * H]; bt = bcat[2 * H:3 * H]
    outs = []
    for s in range(L):
        xs = x[:, s, :]
        z = torch.cat([h, xs], dim=-1)
        ff1 = torch.tanh(z @ Wf1 + bf1)
        ff2 = torch.tanh(z @ Wf2 + bf2)
        g = torch.sigmoid(z @ Wt + bt)
        h = ff1 * (1 - g) + ff2 * g
        outs.append(h)
    return torch.stack(outs, dim=1)


def torch_encoder(u0, Ws, Bs, H):
    a = u0
    for layer in range(len(Ws)):
        a = torch_cfc_loop(a, Ws[layer], Bs[layer], H)
    return a


def fused_encoder(u0, Ws, Bs, H):
    a = u0
    for layer in range(len(Ws)):
        a = custom_cfc(a, Ws[layer], Bs[layer])
    return a


def bench(fn, warmup=3, repeat=20):
    with torch.no_grad():
        for _ in range(warmup): fn()
        torch.npu.synchronize(); ts = []
        for _ in range(repeat):
            t = time.perf_counter(); fn(); torch.npu.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 336
    enc_in = int(sys.argv[3]) if len(sys.argv) > 3 else 11
    H = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    layers = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    torch.manual_seed(0)
    print(f"[shape] B={B} L={L} enc_in={enc_in} H={H} layers={layers}", flush=True)

    series = torch.randn(B, L, enc_in, device=DEV)
    series = (series - series.mean(dim=1, keepdim=True)) / (series.std(dim=1, keepdim=True) + 1e-6)
    u0 = series.contiguous()

    stdv = 1.0 / (H ** 0.5)
    Ws, Bs = [], []
    for layer in range(layers):
        IN = enc_in if layer == 0 else H
        Din = H + IN
        Ws.append(((torch.rand(Din, 3 * H, device=DEV) * 2 - 1) * stdv).contiguous())
        Bs.append(((torch.rand(3 * H, device=DEV) * 2 - 1) * stdv).contiguous())

    with torch.no_grad():
        a_ref = u0
        for layer in range(layers):
            t_out = torch_cfc_loop(a_ref, Ws[layer], Bs[layer], H)
            o_out = custom_cfc(a_ref, Ws[layer], Bs[layer])
            dl = (o_out - t_out).abs().max().item()
            rl = dl / (t_out.abs().max().item() + 1e-9)
            print(f"[layer{layer}] op vs torch-loop  max_diff={dl:.3e}  rel={rl:.3e}", flush=True)
            a_ref = t_out

        ref = torch_encoder(u0, Ws, Bs, H)
        fus = fused_encoder(u0, Ws, Bs, H)
        d = (fus - ref).abs().max().item()
        rel = d / (ref.abs().max().item() + 1e-9)
        print(f"[correct] fused-encoder vs fp32-torch  max_diff={d:.3e}  rel={rel:.3e}", flush=True)

        ms_torch_1 = bench(lambda: torch_cfc_loop(u0, Ws[0], Bs[0], H), warmup=2, repeat=10)
        ms_fused_1 = bench(lambda: custom_cfc(u0, Ws[0], Bs[0]), warmup=3, repeat=20)

        ms_torch_e = bench(lambda: torch_encoder(u0, Ws, Bs, H), warmup=2, repeat=6)
        ms_fused_e = bench(lambda: fused_encoder(u0, Ws, Bs, H), warmup=3, repeat=12)

    res = {"shape": {"B": B, "L": L, "enc_in": enc_in, "H": H, "layers": layers},
           "correctness_max_diff": d, "correctness_rel": rel,
           "component_torch_ms": ms_torch_1, "component_fused_ms": ms_fused_1,
           "component_speedup": ms_torch_1 / ms_fused_1,
           "e2e_torch_ms": ms_torch_e, "e2e_fused_ms": ms_fused_e,
           "e2e_speedup": ms_torch_e / ms_fused_e}
    print("[RESULT] " + json.dumps(res), flush=True)
    with open("/home/ql2025/work/tslib_cann_ops_dev/docs/cfc_scan_fused_e2e_results.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
