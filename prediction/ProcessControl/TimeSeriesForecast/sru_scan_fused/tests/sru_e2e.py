#!/usr/bin/env python3
"""SRU encoder E2E: per-timestep loop vs SruScanFused."""
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
ROOT = "/home/ql2025/work/tslib_cann_ops_dev/build/msopgen_sru_scan_fused/build_out"
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
    op.aclnnSruScanFusedGetWorkspaceSize.argtypes = [c_void_p]*4 + [POINTER(c_uint64), POINTER(c_void_p)]
    op.aclnnSruScanFusedGetWorkspaceSize.restype = c_int
    op.aclnnSruScanFused.argtypes = [c_void_p, c_uint64, c_void_p, c_void_p]
    op.aclnnSruScanFused.restype = c_int
    return acl, op

_ACL, _OP = init_op()

def _t(t):
    dims = list(t.shape); strides = []; s = 1
    for d in reversed(dims): strides.insert(0, s); s *= d
    da = (c_int64 * len(dims))(*dims); sa = (c_int64 * len(strides))(*strides)
    return _ACL.aclCreateTensor(da, len(dims), ACL_FLOAT, sa, 0, ACL_FORMAT_ND, da, len(dims), c_void_p(t.data_ptr()))

def custom_op(x, w, b):
    x=x.contiguous(); w=w.contiguous(); b=b.contiguous()
    B,L,IN=x.shape; H=w.shape[1]
    out=torch.zeros(B,L,H,device=DEV)
    desc=[_t(x),_t(w),_t(b),_t(out)]
    ws=c_uint64(0); ex=c_void_p()
    _OP.aclnnSruScanFusedGetWorkspaceSize(desc[0],desc[1],desc[2],desc[3],byref(ws),byref(ex))
    wbuf=torch.empty((int(ws.value),),dtype=torch.uint8,device=DEV) if ws.value else None
    _OP.aclnnSruScanFused(c_void_p(0 if wbuf is None else wbuf.data_ptr()),ws.value,ex,c_void_p(torch.npu.current_stream().npu_stream))
    torch.npu.synchronize()
    for d in desc: _ACL.aclDestroyTensor(d)
    return out

def torch_loop(x, w, b, H):
    B,L,IN=x.shape
    W=w[0:IN,:]; Wf=w[IN:2*IN,:]; Wr=w[2*IN:3*IN,:]
    vf=b[0:H]; vr=b[H:2*H]; bf=b[2*H:3*H]; br=b[3*H:4*H]
    c=torch.zeros(B,H,device=x.device); outs=[]
    for s in range(L):
        xs=x[:,s,:]
        xt=xs@W
        f=torch.sigmoid(xs@Wf + vf*c + bf)
        r=torch.sigmoid(xs@Wr + vr*c + br)
        c=f*c+(1-f)*xt
        h=r*torch.tanh(c)+(1-r)*xt
        outs.append(h)
    return torch.stack(outs,dim=1)

def torch_enc(u0,Ws,Bs,H):
    a=u0
    for i in range(len(Ws)): a=torch_loop(a,Ws[i],Bs[i],H)
    return a

def fused_enc(u0,Ws,Bs,H):
    a=u0
    for i in range(len(Ws)): a=custom_op(a,Ws[i],Bs[i])
    return a

def bench(fn, warmup=3, repeat=20):
    with torch.no_grad():
        for _ in range(warmup): fn()
        torch.npu.synchronize(); ts=[]
        for _ in range(repeat):
            t=time.perf_counter(); fn(); torch.npu.synchronize()
            ts.append((time.perf_counter()-t)*1000)
    return float(np.median(ts))

def main():
    B=int(sys.argv[1]) if len(sys.argv)>1 else 32
    L=int(sys.argv[2]) if len(sys.argv)>2 else 336
    enc_in=int(sys.argv[3]) if len(sys.argv)>3 else 11
    H=int(sys.argv[4]) if len(sys.argv)>4 else 64
    layers=int(sys.argv[5]) if len(sys.argv)>5 else 3
    torch.manual_seed(0)
    print(f"[shape] B={B} L={L} enc_in={enc_in} H={H} layers={layers}",flush=True)
    series=torch.randn(B,L,enc_in,device=DEV)
    series=(series-series.mean(dim=1,keepdim=True))/(series.std(dim=1,keepdim=True)+1e-6)
    u0=series.contiguous()
    stdv=1.0/(H**0.5)
    Ws,Bs=[],[]
    for layer in range(layers):
        IN=enc_in if layer==0 else H
        w=((torch.rand(3*IN,H,device=DEV)*2-1)*stdv).contiguous()
        b=((torch.rand(4*H,device=DEV)*2-1)*stdv).contiguous()
        Ws.append(w); Bs.append(b)

    with torch.no_grad():
        a_ref=u0
        for layer in range(layers):
            t_out=torch_loop(a_ref,Ws[layer],Bs[layer],H)
            o_out=custom_op(a_ref,Ws[layer],Bs[layer])
            dl=(o_out-t_out).abs().max().item()
            rl=dl/(t_out.abs().max().item()+1e-9)
            print(f"[layer{layer}] op vs torch-loop  max_diff={dl:.3e}  rel={rl:.3e}",flush=True)
            a_ref=t_out
        ref=torch_enc(u0,Ws,Bs,H); fus=fused_enc(u0,Ws,Bs,H)
        d=(fus-ref).abs().max().item(); rel=d/(ref.abs().max().item()+1e-9)
        print(f"[correct] fused-encoder vs fp32-torch  max_diff={d:.3e}  rel={rel:.3e}",flush=True)
        ms1t=bench(lambda: torch_loop(u0,Ws[0],Bs[0],H),2,10)
        ms1f=bench(lambda: custom_op(u0,Ws[0],Bs[0]),3,20)
        mset=bench(lambda: torch_enc(u0,Ws,Bs,H),2,6)
        msef=bench(lambda: fused_enc(u0,Ws,Bs,H),3,12)

    res={"shape":{"B":B,"L":L,"enc_in":enc_in,"H":H,"layers":layers},
         "correctness_max_diff":d,"correctness_rel":rel,
         "component_torch_ms":ms1t,"component_fused_ms":ms1f,"component_speedup":ms1t/ms1f,
         "e2e_torch_ms":mset,"e2e_fused_ms":msef,"e2e_speedup":mset/msef}
    print("[RESULT] "+json.dumps(res),flush=True)

if __name__=="__main__":
    main()
