#!/usr/bin/env python3
"""S6 Selective Scan Fused - Correctness probe (Mamba-style)"""
import ctypes, glob
from ctypes import POINTER, byref, c_int, c_int64, c_uint64, c_void_p
import importlib

import numpy as np
import torch

importlib.import_module("torch_npu")
DEV = "npu:0"
ROOT = "/home/ql2025/work/tslib_cann_ops_dev/build/msopgen_s6scan_fused/build_out"
ACL_FLOAT, ACL_FORMAT_ND = 0, 2

def init_op():
    torch.empty(1, device=DEV); torch.npu.synchronize()
    for so in glob.glob(f"{ROOT}/**/libascend_all_ops.so", recursive=True):
        ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL); break
    ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    op = ctypes.CDLL(f"{ROOT}/op_api/lib/libcust_opapi.so", mode=ctypes.RTLD_GLOBAL)
    acl = ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    acl.aclCreateTensor.argtypes = [POINTER(c_int64), c_uint64, c_int, POINTER(c_int64), c_int64, c_int, POINTER(c_int64), c_uint64, c_void_p]
    acl.aclCreateTensor.restype = c_void_p
    acl.aclDestroyTensor.argtypes = [c_void_p]; acl.aclDestroyTensor.restype = c_int
    op.aclnnS6scanFusedGetWorkspaceSize.argtypes = [c_void_p]*4 + [POINTER(c_uint64), POINTER(c_void_p)]
    op.aclnnS6scanFusedGetWorkspaceSize.restype = c_int
    op.aclnnS6scanFused.argtypes = [c_void_p, c_uint64, c_void_p, c_void_p]
    op.aclnnS6scanFused.restype = c_int
    return acl, op

_ACL, _OP = init_op()

def _t(t):
    dims = list(t.shape); strides = []; s = 1
    for d in reversed(dims): strides.insert(0, s); s *= d
    da = (c_int64 * len(dims))(*dims); sa = (c_int64 * len(strides))(*strides)
    return _ACL.aclCreateTensor(da, len(dims), ACL_FLOAT, sa, 0, ACL_FORMAT_ND, da, len(dims), c_void_p(t.data_ptr()))

def npu_s6(x, w, b):
    x = x.contiguous(); w = w.contiguous(); b = b.contiguous()
    B, L, IN = x.shape; H = w.shape[1]
    out = torch.zeros(B, L, H, device=DEV)
    desc = [_t(x), _t(w), _t(b), _t(out)]
    ws = c_uint64(0); ex = c_void_p()
    ret = _OP.aclnnS6scanFusedGetWorkspaceSize(desc[0], desc[1], desc[2], desc[3], byref(ws), byref(ex))
    assert ret == 0, f"GetWorkspaceSize ret={ret}"
    wbuf = torch.empty((int(ws.value),), dtype=torch.uint8, device=DEV) if ws.value else None
    ret2 = _OP.aclnnS6scanFused(c_void_p(0 if wbuf is None else wbuf.data_ptr()), ws.value, ex, c_void_p(torch.npu.current_stream().npu_stream))
    assert ret2 == 0, f"Execute ret={ret2}"
    torch.npu.synchronize()
    for d in desc: _ACL.aclDestroyTensor(d)
    return out

def cpu_s6(x_np, w_np, b_np, B, L, IN, H):
    # Weight: (3*IN, H) = Wd(IN,H) + Wb(IN,H) + Wx(IN,H)
    # Bias: (4*H,) = bd(H) + bb(H) + bx(H) + a_param(H)
    Wd = w_np[:IN, :]
    Wb = w_np[IN:2*IN, :]
    Wx = w_np[2*IN:, :]
    bd = b_np[:H]
    bb = b_np[H:2*H]
    bx = b_np[2*H:3*H]
    a_param = b_np[3*H:4*H]
    out = np.zeros((B, L, H), dtype=np.float32)
    for bi in range(B):
        h = np.zeros(H, dtype=np.float32)
        for t in range(L):
            xt = x_np[bi, t, :]
            # delta = softplus(Wd*x + bd)
            delta_pre = xt @ Wd + bd
            delta = np.log(1.0 + np.exp(delta_pre))
            # A_disc = exp(-delta * a_param)
            A_disc = np.exp(-delta * a_param)
            # B_gate = sigmoid(Wb*x + bb)
            B_gate = 1.0 / (1.0 + np.exp(-(xt @ Wb + bb)))
            # x_proj = Wx*x + bx
            x_proj = xt @ Wx + bx
            # h = A_disc * h + B_gate * x_proj
            h = A_disc * h + B_gate * x_proj
            out[bi, t, :] = h
    return out

shapes = [(4,7,16,16), (8,33,21,32), (16,64,11,64), (8,50,64,64)]
np.random.seed(42)

for B, L, IN, H in shapes:
    x_np = ((np.arange(B*L*IN) % 17 - 8) * 0.003).astype(np.float32).reshape(B, L, IN)
    w_np = ((np.arange(IN*3*H) % 13 - 6) * 0.001).astype(np.float32).reshape(IN*3, H)
    b_np = ((np.arange(4*H) % 11 - 5) * 0.001).astype(np.float32)
    # Make a_param positive (required for stable decay)
    b_np[3*H:4*H] = np.abs(b_np[3*H:4*H]) + 0.1

    ref = cpu_s6(x_np, w_np, b_np, B, L, IN, H)
    x_t = torch.from_numpy(x_np).to(DEV)
    w_t = torch.from_numpy(w_np).to(DEV)
    b_t = torch.from_numpy(b_np).to(DEV)
    npu_out = npu_s6(x_t, w_t, b_t).cpu().numpy()
    max_diff = np.abs(ref - npu_out).max()
    print(f"CHECK B={B} L={L} IN={IN} H={H} max_diff={max_diff:.6e}")
