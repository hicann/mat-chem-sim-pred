#!/usr/bin/env python3
"""TiRex block0 sLSTM E2E: replace torch recurrent scan with custom NPU op.

Loads real NX-AI/TiRex, builds real hidden from MTO data, extracts the sLSTM
cell tensors (recurrent kernel, bias, gate projections), runs:
  (1) correctness: custom-op h-sequence vs fp32 torch reference scan (same math)
  (2) latency: torch cell scan (cell_impl) vs custom op; and full sLSTM layer
      (gate-proj + scan + group_norm) torch vs custom.
"""
import ctypes
import glob
import importlib
import json
import time
from ctypes import POINTER, byref, c_int, c_int64, c_uint64, c_void_p

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tirex

importlib.import_module("torch_npu")

DEV = "npu:0"
ROOT = "/home/ql2025/work/tslib_cann_ops_dev/build/msopgen_tirex_slstm_cell/build_out"
DATA_PATH = "/root/sixseven/knowledge_engineer/MTO_cleaned_116_merged.csv"
TARGET_COL = "MTO_13-AI-11403F.PV"
ACL_FLOAT, ACL_FORMAT_ND = 0, 2


def init_op():
    torch.empty(1, device=DEV); torch.npu.synchronize()
    for so in glob.glob(f"{ROOT}/**/libascend_all_ops.so", recursive=True):
        ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL); break
    acl = ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    op = ctypes.CDLL(f"{ROOT}/op_api/lib/libcust_opapi.so", mode=ctypes.RTLD_GLOBAL)
    acl.aclCreateTensor.argtypes = [POINTER(c_int64), c_uint64, c_int, POINTER(c_int64), c_int64,
                                    c_int, POINTER(c_int64), c_uint64, c_void_p]
    acl.aclCreateTensor.restype = c_void_p
    acl.aclDestroyTensor.argtypes = [c_void_p]; acl.aclDestroyTensor.restype = c_int
    op.aclnnTirexSlstmCellGetWorkspaceSize.argtypes = [c_void_p]*6 + [POINTER(c_uint64), POINTER(c_void_p)]
    op.aclnnTirexSlstmCellGetWorkspaceSize.restype = c_int
    op.aclnnTirexSlstmCell.argtypes = [c_void_p, c_uint64, c_void_p, c_void_p]
    op.aclnnTirexSlstmCell.restype = c_int
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


def custom_cell(inp, R, bias, init):
    """inp[B,S,4H] R[heads,hd,4hd] bias[4H] init[4,B,H] -> out[B,S,H], final[4,B,H] (fp32, npu)."""
    inp = inp.contiguous(); R = R.contiguous(); bias = bias.contiguous(); init = init.contiguous()
    B, S, fourH = inp.shape; H = fourH // 4
    out = torch.zeros(B, S, H, device=DEV); final = torch.zeros(4, B, H, device=DEV)
    desc = [_t(inp), _t(R), _t(bias), _t(init), _t(out), _t(final)]
    ws = c_uint64(0); ex = c_void_p()
    r = _OP.aclnnTirexSlstmCellGetWorkspaceSize(desc[0], desc[1], desc[2], desc[3], desc[4], desc[5], byref(ws), byref(ex))
    if r: raise RuntimeError(f"GetWorkspaceSize ret={r}")
    wbuf = torch.empty((int(ws.value),), dtype=torch.uint8, device=DEV) if ws.value else None
    r = _OP.aclnnTirexSlstmCell(c_void_p(0 if wbuf is None else wbuf.data_ptr()), ws.value, ex,
                                c_void_p(torch.npu.current_stream().npu_stream))
    if r: raise RuntimeError(f"aclnnTirexSlstmCell ret={r}")
    torch.npu.synchronize()
    for d in desc: _ACL.aclDestroyTensor(d)
    return out, final


def ref_scan(inp, R, bias, heads):
    """fp32 torch reference scan matching the op math. inp[B,S,4H] order [i,f,z,o]."""
    B, S, fourH = inp.shape; H = fourH // 4; hd = H // heads
    c = torch.zeros(B, H, device=DEV); n = torch.zeros(B, H, device=DEV)
    m = torch.zeros(B, H, device=DEV); h = torch.zeros(B, H, device=DEV)
    outs = []
    for s in range(S):
        ry = torch.einsum('bhk,hkj->bhj', h.view(B, heads, hd), R)      # [B,heads,4hd]
        ry = ry.view(B, heads, 4, hd).permute(0, 2, 1, 3).reshape(B, 4 * H)  # [i,f,z,o] gate-major
        raw = (inp[:, s, :] + ry + bias).view(B, 4, H)
        iraw, fraw, zraw, oraw = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        logfplusm = m + F.logsigmoid(torch.clamp(fraw, max=15.0))
        mnew = iraw if s == 0 else torch.maximum(iraw, logfplusm)
        igate = torch.minimum(torch.exp(iraw - mnew), torch.ones_like(iraw))
        fgate = torch.minimum(torch.exp(logfplusm - mnew), torch.ones_like(iraw))
        zgate = torch.tanh(zraw); ogate = torch.sigmoid(oraw)
        cnew = fgate * c + igate * zgate; nnew = fgate * n + igate
        denom = torch.where(nnew == 0.0, torch.full_like(nnew, 1e-6), nnew)
        hnew = ogate * cnew / denom
        c, n, m, h = cnew, nnew, mnew, hnew
        outs.append(hnew.unsqueeze(1))
    return torch.cat(outs, dim=1)  # [B,S,H]


def make_setup():
    df = pd.read_csv(DATA_PATH); y = df[TARGET_COL].to_numpy(np.float32)
    y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-6); rows = []
    for s in range(0, len(y) - 524, 12):
        w = y[s:s + 512]
        if np.isfinite(w).all(): rows.append(w.astype(np.float32))
        if len(rows) >= 64: break
    ctx = torch.tensor(np.stack(rows), device=DEV)
    m = tirex.load_model('NX-AI/TiRex', device=DEV, backend='torch').eval()
    ctx, _ = m._adjust_context_length(ctx, m.config.train_ctx_len, m.config.train_ctx_len)
    tok, st = m.tokenizer.input_transform(ctx)
    mask = torch.isnan(tok).logical_not().to(tok.dtype)
    tok = torch.nan_to_num(tok, nan=m.config.nan_mask_value)
    inp = torch.cat((tok, mask), dim=2); hidden = m.input_patch_embedding(inp)
    return m, hidden


def bench(fn, warmup=2, repeat=10):
    with torch.no_grad():
        for _ in range(warmup): fn()
        torch.npu.synchronize(); ts = []
        for _ in range(repeat):
            t = time.perf_counter(); fn(); torch.npu.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def main():
    m, hidden = make_setup()
    block = m.blocks[0]; layer = block.slstm_layer; cell = layer.slstm_cell
    B, S, D = hidden.shape
    heads = cell.config.num_heads; gates = cell.config.num_gates; hd = cell.config.head_dim
    H = heads * hd
    x = block.norm_slstm(hidden)
    print(f"[shape] B={B} S={S} D={D} H={H} heads={heads} gates={gates} hd={hd}", flush=True)

    with torch.no_grad():
        ig = layer.igate(x).float(); fg = layer.fgate(x).float()
        zg = layer.zgate(x).float(); og = layer.ogate(x).float()
        inp = torch.cat((ig, fg, zg, og), dim=-1).contiguous()              # [B,S,4H] order i,f,z,o
        R = cell._recurrent_kernel_.float().contiguous()                    # [heads,hd,4hd]
        bias = cell._bias_.reshape(heads, gates, hd).permute(1, 0, 2).reshape(-1).float().contiguous()  # [4H]
        init = torch.zeros(4, B, H, device=DEV)
        print(f"[tensors] inp={list(inp.shape)} R={list(R.shape)} bias={list(bias.shape)}", flush=True)

        out_c, final_c = custom_cell(inp, R, bias, init)
        out_ref = ref_scan(inp, R, bias, heads)
        d = (out_c - out_ref).abs().max().item()
        rel = d / (out_ref.abs().max().item() + 1e-9)
        print(f"[correct] custom-op vs fp32-torch-ref  max_diff={d:.3e}  rel={rel:.3e}", flush=True)

        ms_cell = bench(lambda: cell._impl_torch(cell._get_input(torch.cat((fg, ig, zg, og), dim=-1)),
                                                 cell._get_state(cell._get_input(torch.cat((fg, ig, zg, og), dim=-1)), None)),
                        warmup=1, repeat=5)
        ms_layer = bench(lambda: layer(x, slstm_state=None), warmup=1, repeat=5)
        ms_op = bench(lambda: custom_cell(inp, R, bias, init), warmup=3, repeat=20)

        def custom_layer():
            ig2 = layer.igate(x); fg2 = layer.fgate(x); zg2 = layer.zgate(x); og2 = layer.ogate(x)
            inp2 = torch.cat((ig2.float(), fg2.float(), zg2.float(), og2.float()), dim=-1).contiguous()
            o, _ = custom_cell(inp2, R, bias, init)
            o4 = o.view(B, S, heads, hd).permute(0, 2, 1, 3).contiguous()  # [B,heads,S,hd]
            return layer.group_norm(o4).transpose(1, 2).reshape(B, S, H)
        ms_clayer = bench(custom_layer, warmup=3, repeat=20)

    res = {"shape": {"B": B, "S": S, "H": H, "heads": heads},
           "correctness_max_diff": d, "correctness_rel": rel,
           "torch_cell_impl_ms": ms_cell, "torch_slstm_layer_ms": ms_layer,
           "custom_op_ms": ms_op, "custom_layer_ms": ms_clayer,
           "speedup_cell": ms_cell / ms_op, "speedup_layer": ms_layer / ms_clayer}
    print("[RESULT] " + json.dumps(res), flush=True)
    with open("/home/ql2025/work/tslib_cann_ops_dev/docs/tirex_slstm_e2e_results.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
