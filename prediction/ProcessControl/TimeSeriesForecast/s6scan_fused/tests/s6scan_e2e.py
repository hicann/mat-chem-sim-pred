#!/usr/bin/env python3
"""S6 Selective Scan Fused - E2E benchmark"""
import ctypes, glob, time, json
from ctypes import POINTER, byref, c_int, c_int64, c_uint64, c_void_p
import importlib

import torch

importlib.import_module("torch_npu")
DEV = "npu:0"
ROOT = "/home/ql2025/work/tslib_cann_ops_dev/build/msopgen_s6scan_fused/build_out"
ACL_FLOAT, ACL_FORMAT_ND = 0, 2

torch.empty(1, device=DEV); torch.npu.synchronize()
for so in glob.glob(f"{ROOT}/**/libascend_all_ops.so", recursive=True):
    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL); break
ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
_OP = ctypes.CDLL(f"{ROOT}/op_api/lib/libcust_opapi.so", mode=ctypes.RTLD_GLOBAL)
_ACL = ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
_ACL.aclCreateTensor.argtypes = [POINTER(c_int64), c_uint64, c_int, POINTER(c_int64), c_int64, c_int, POINTER(c_int64), c_uint64, c_void_p]
_ACL.aclCreateTensor.restype = c_void_p
_ACL.aclDestroyTensor.argtypes = [c_void_p]; _ACL.aclDestroyTensor.restype = c_int
_OP.aclnnS6scanFusedGetWorkspaceSize.argtypes = [c_void_p]*4 + [POINTER(c_uint64), POINTER(c_void_p)]
_OP.aclnnS6scanFusedGetWorkspaceSize.restype = c_int
_OP.aclnnS6scanFused.argtypes = [c_void_p, c_uint64, c_void_p, c_void_p]
_OP.aclnnS6scanFused.restype = c_int

def _t(t):
    dims = list(t.shape); strides = []; s = 1
    for d in reversed(dims): strides.insert(0, s); s *= d
    da = (c_int64 * len(dims))(*dims); sa = (c_int64 * len(strides))(*strides)
    return _ACL.aclCreateTensor(da, len(dims), ACL_FLOAT, sa, 0, ACL_FORMAT_ND, da, len(dims), c_void_p(t.data_ptr()))

def custom_op(x, w, b):
    x = x.contiguous(); w = w.contiguous(); b = b.contiguous()
    B, L, IN = x.shape; H = w.shape[1]
    out = torch.zeros(B, L, H, device=DEV)
    desc = [_t(x), _t(w), _t(b), _t(out)]
    ws = c_uint64(0); ex = c_void_p()
    _OP.aclnnS6scanFusedGetWorkspaceSize(desc[0], desc[1], desc[2], desc[3], byref(ws), byref(ex))
    wbuf = torch.empty((int(ws.value),), dtype=torch.uint8, device=DEV) if ws.value else None
    _OP.aclnnS6scanFused(c_void_p(0 if wbuf is None else wbuf.data_ptr()), ws.value, ex, c_void_p(torch.npu.current_stream().npu_stream))
    torch.npu.synchronize()
    for d in desc: _ACL.aclDestroyTensor(d)
    return out

def torch_s6(x, w, b):
    B, L, IN = x.shape; H = w.shape[1]
    Wd = w[:IN, :]; Wb = w[IN:2*IN, :]; Wx = w[2*IN:, :]
    bd = b[:H]; bb = b[H:2*H]; bx = b[2*H:3*H]; a_param = b[3*H:4*H]
    h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    outs = []
    for t in range(L):
        xt = x[:, t, :]
        delta = torch.nn.functional.softplus(xt @ Wd + bd)
        A_disc = torch.exp(-delta * a_param)
        B_gate = torch.sigmoid(xt @ Wb + bb)
        x_proj = xt @ Wx + bx
        h = A_disc * h + B_gate * x_proj
        outs.append(h.unsqueeze(1))
    return torch.cat(outs, dim=1)

B, L, IN, H, layers = 32, 336, 11, 64, 3
print(f"[shape] B={B} L={L} enc_in={IN} H={H} layers={layers}")
torch.manual_seed(42)
ws = [torch.randn(IN*3 if i==0 else H*3, H, device=DEV)*0.01 for i in range(layers)]
bs = [torch.cat([torch.randn(3*H, device=DEV)*0.01, torch.rand(H, device=DEV)*0.5+0.1]) for _ in range(layers)]
x = torch.randn(B, L, IN, device=DEV)*0.1

out_fused = x; out_torch = x
for i in range(layers):
    out_fused = custom_op(out_fused, ws[i], bs[i])
    out_torch = torch_s6(out_torch, ws[i], bs[i])
md_final = (out_fused - out_torch).abs().max().item()
print(f"[correct] fused-encoder vs fp32-torch  max_diff={md_final:.3e}")

torch.npu.synchronize()
w0 = ws[0]; b0 = bs[0]
for _ in range(3): custom_op(x, w0, b0)
torch.npu.synchronize()
t0 = time.time()
for _ in range(10): custom_op(x, w0, b0)
torch.npu.synchronize()
fused_ms = (time.time()-t0)/10*1000

for _ in range(3): torch_s6(x, w0, b0)
torch.npu.synchronize()
t0 = time.time()
for _ in range(10): torch_s6(x, w0, b0)
torch.npu.synchronize()
torch_ms = (time.time()-t0)/10*1000

for _ in range(3):
    tmp = x
    for i in range(layers): tmp = custom_op(tmp, ws[i], bs[i])
torch.npu.synchronize()
t0 = time.time()
for _ in range(10):
    tmp = x
    for i in range(layers): tmp = custom_op(tmp, ws[i], bs[i])
torch.npu.synchronize()
e2e_fused = (time.time()-t0)/10*1000

for _ in range(3):
    tmp = x
    for i in range(layers): tmp = torch_s6(tmp, ws[i], bs[i])
torch.npu.synchronize()
t0 = time.time()
for _ in range(10):
    tmp = x
    for i in range(layers): tmp = torch_s6(tmp, ws[i], bs[i])
torch.npu.synchronize()
e2e_torch = (time.time()-t0)/10*1000

result = {"correctness_max_diff": md_final, "component_torch_ms": round(torch_ms,2), "component_fused_ms": round(fused_ms,2), "component_speedup": round(torch_ms/fused_ms,2), "e2e_torch_ms": round(e2e_torch,2), "e2e_fused_ms": round(e2e_fused,2), "e2e_speedup": round(e2e_torch/e2e_fused,2)}
print(f"[RESULT] {json.dumps(result)}")
