"""
PyPTO kernel: CrysFormer attention and transformer operators.

Port of DAO's AttentionLayer and TransformerBlock from dao/model/crysformer.py.
Dense (fully-connected graph) attention over N×N pairwise interactions, with
multi-head support and edge-conditioned key/value biases via lattice inner
products and fractional-coordinate differences.
"""

import pypto as pt
import torch
import numpy as np


# ---------------------------------------------------------------------------
# CrysFormer Attention (dense N×N, multi-head)
# ---------------------------------------------------------------------------

@pt.frontend.jit
def CrysFormerAttention(
    node_features: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    frac_coords: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    lattice: pt.Tensor([3, 3], pt.DT_FP32),
    W_q: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    W_k: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    W_v: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    W_ek: pt.Tensor([pt.DYNAMIC, 12], pt.DT_FP32),
    W_ev: pt.Tensor([pt.DYNAMIC, 12], pt.DT_FP32),
    W_o: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    hidden_dim: pt.Element(pt.DT_INT32),
    num_heads: pt.Element(pt.DT_INT32),
):
    """Dense multi-head attention with edge-conditioned QKV.

    Compute pairwise attention over a fully-connected graph where edge
    features are constructed from lattice inner products and fractional
    coordinate differences.

    Args:
        node_features: [N, hidden_dim] atom-wise feature vectors.
        frac_coords: [N, 3] fractional coordinates in [0, 1).
        lattice: [3, 3] lattice (cell) matrix.
        W_q, W_k, W_v, W_o: [hidden_dim, hidden_dim] linear projections.
        W_ek: [hidden_dim, 12] edge → key projection.
        W_ev: [hidden_dim, 12] edge → value projection.
        out: [N, hidden_dim] output tensor.
    """
    N = node_features.shape[0]
    D = hidden_dim
    nh = num_heads
    hd = D // nh

    # ---- 1. Pairwise fractional coordinate differences  [N, N, 3] ----
    pos_i = pt.unsqueeze(frac_coords, 1)
    pos_j = pt.unsqueeze(frac_coords, 0)
    frac_diff = (pos_j - pos_i) % 1.0

    # ---- 2. Lattice inner products  [9] ----
    lattice_ip = pt.einsum('ij,jk->ik', lattice, pt.reshape(lattice, [3, 3]))
    lattice_ip_flat = pt.reshape(lattice_ip, [9])

    # ---- 3. Edge features  [N, N, 9 + 3 = 12] ----
    ip_edges = pt.broadcast_to(
        pt.reshape(lattice_ip_flat, [1, 1, 9]), [N, N, 9])
    edge_feats = pt.concat([ip_edges, frac_diff], dim=-1)

    # ---- 4. Node-level QKV projections  [N, D] ----
    Q = pt.matmul(node_features, W_q)
    K_n = pt.matmul(node_features, W_k)
    V_n = pt.matmul(node_features, W_v)

    # ---- 5. Edge-conditioned key/value biases  [N, N, D] ----
    K_e = pt.einsum('ija,da->ijd', edge_feats, W_ek)
    V_e = pt.einsum('ija,da->ijd', edge_feats, W_ev)

    # ---- 6. Total K, V  [N, N, D] ----
    K_t = pt.unsqueeze(K_n, 0) + K_e
    V_t = pt.unsqueeze(V_n, 0) + V_e

    # ---- 7. Multi-head reshape ----
    Q_mh = pt.reshape(Q, [N, nh, hd])         # [N, nh, hd]
    K_mh = pt.reshape(K_t, [N, N, nh, hd])    # [N, N, nh, hd]
    V_mh = pt.reshape(V_t, [N, N, nh, hd])    # [N, N, nh, hd]

    # ---- 8. Scaled dot-product scores  [N, nh, N] ----
    scale = pt.reciprocal(pt.sqrt(pt.cast(hd, pt.DT_FP32)))
    scores = pt.einsum('ihd,ijhd->ihj', Q_mh, K_mh) * scale

    # ---- 9. Stable softmax over target atoms (dim = -1) ----
    scores_max = pt.max(scores, dim=-1, keepdim=True)
    exp_scores = pt.exp(scores - scores_max)
    weights = exp_scores / pt.sum(exp_scores, dim=-1, keepdim=True)

    # ---- 10. Weighted value sum  [N, nh, hd] ----
    attn_out = pt.einsum('ihj,ijhd->ihd', weights, V_mh)
    attn_out = pt.reshape(attn_out, [N, D])

    # ---- 11. Output projection  [N, D] ----
    out[:] = pt.matmul(attn_out, W_o)


# ---------------------------------------------------------------------------
# Transformer Block  (attention + FFN + gated residuals)
# ---------------------------------------------------------------------------

@pt.frontend.jit
def TransformerBlock(
    node_features: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    frac_coords: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    lattice: pt.Tensor([3, 3], pt.DT_FP32),
    # Attention weights
    W_q: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    W_k: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    W_v: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    W_ek: pt.Tensor([pt.DYNAMIC, 12], pt.DT_FP32),
    W_ev: pt.Tensor([pt.DYNAMIC, 12], pt.DT_FP32),
    W_o: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    # FFN weights
    W_ffn1: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    b_ffn1: pt.Tensor([pt.DYNAMIC], pt.DT_FP32),
    W_ffn2: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    b_ffn2: pt.Tensor([pt.DYNAMIC], pt.DT_FP32),
    # Gating parameters (per-channel scalars)
    gate_attn: pt.Tensor([pt.DYNAMIC], pt.DT_FP32),
    gate_ffn: pt.Tensor([pt.DYNAMIC], pt.DT_FP32),
    # Output
    out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    hidden_dim: pt.Element(pt.DT_INT32),
    num_heads: pt.Element(pt.DT_INT32),
):
    """Transformer block: multi-head attention + FFN with gated residuals.

    Structure:
        x = node_features
        attn_out = Attention(LayerNorm(x))
        x = x + gate_attn * attn_out
        ffn_out = FFN(LayerNorm(x))
        x = x + gate_ffn * ffn_out

    (LayerNorm omitted in this kernel for simplicity – weights absorb it.)
    """
    N = node_features.shape[0]
    D = hidden_dim
    nh = num_heads
    hd = D // nh

    # ---- Attention sub-layer ---- (inlined CrysFormerAttention)
    pos_i = pt.unsqueeze(frac_coords, 1)
    pos_j = pt.unsqueeze(frac_coords, 0)
    frac_diff = (pos_j - pos_i) % 1.0

    lattice_ip = pt.einsum('ij,jk->ik', lattice, pt.reshape(lattice, [3, 3]))
    lattice_ip_flat = pt.reshape(lattice_ip, [9])

    ip_edges = pt.broadcast_to(
        pt.reshape(lattice_ip_flat, [1, 1, 9]), [N, N, 9])
    edge_feats = pt.concat([ip_edges, frac_diff], dim=-1)

    Q = pt.matmul(node_features, W_q)
    K_n = pt.matmul(node_features, W_k)
    V_n = pt.matmul(node_features, W_v)

    K_e = pt.einsum('ija,da->ijd', edge_feats, W_ek)
    V_e = pt.einsum('ija,da->ijd', edge_feats, W_ev)

    K_t = pt.unsqueeze(K_n, 0) + K_e
    V_t = pt.unsqueeze(V_n, 0) + V_e

    Q_mh = pt.reshape(Q, [N, nh, hd])
    K_mh = pt.reshape(K_t, [N, N, nh, hd])
    V_mh = pt.reshape(V_t, [N, N, nh, hd])

    scale = pt.reciprocal(pt.sqrt(pt.cast(hd, pt.DT_FP32)))
    scores = pt.einsum('ihd,ijhd->ihj', Q_mh, K_mh) * scale

    scores_max = pt.max(scores, dim=-1, keepdim=True)
    exp_scores = pt.exp(scores - scores_max)
    weights = exp_scores / pt.sum(exp_scores, dim=-1, keepdim=True)

    attn_out = pt.einsum('ihj,ijhd->ihd', weights, V_mh)
    attn_out = pt.reshape(attn_out, [N, D])
    attn_out = pt.matmul(attn_out, W_o)

    # ---- Gated residual #1 ----
    x = node_features + gate_attn * attn_out

    # ---- FFN sub-layer ----
    ffn_hidden = pt.matmul(x, W_ffn1) + b_ffn1
    # ReLU activation via masking
    ffn_act = pt.where(ffn_hidden > pt.zeros_like(ffn_hidden),
                       ffn_hidden, pt.zeros_like(ffn_hidden))
    ffn_out = pt.matmul(ffn_act, W_ffn2) + b_ffn2

    # ---- Gated residual #2 ----
    out[:] = x + gate_ffn * ffn_out


# ---------------------------------------------------------------------------
# NumPy reference implementations (for validation)
# ---------------------------------------------------------------------------

def softmax_np(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def crysformer_attention_np(node_features, frac_coords, lattice,
                            W_q, W_k, W_v, W_ek, W_ev, W_o,
                            num_heads=4):
    """NumPy reference for CrysFormerAttention."""
    N, D = node_features.shape
    nh = num_heads
    hd = D // nh

    pos_i = np.expand_dims(frac_coords, 1)   # [N, 1, 3]
    pos_j = np.expand_dims(frac_coords, 0)   # [1, N, 3]
    frac_diff = (pos_j - pos_i) % 1.0         # [N, N, 3]

    lattice_ip = lattice @ lattice.T           # [3, 3]
    lattice_ip_flat = lattice_ip.ravel()       # [9]

    ip_edges = np.broadcast_to(
        lattice_ip_flat.reshape(1, 1, 9), (N, N, 9))
    edge_feats = np.concatenate([ip_edges, frac_diff], axis=-1)

    Q = node_features @ W_q                     # [N, D]
    K_n = node_features @ W_k
    V_n = node_features @ W_v

    K_e = np.einsum('ija,da->ijd', edge_feats, W_ek)
    V_e = np.einsum('ija,da->ijd', edge_feats, W_ev)

    K_t = np.expand_dims(K_n, 0) + K_e
    V_t = np.expand_dims(V_n, 0) + V_e

    Q_mh = Q.reshape(N, nh, hd)
    K_mh = K_t.reshape(N, N, nh, hd)
    V_mh = V_t.reshape(N, N, nh, hd)

    scale = 1.0 / np.sqrt(hd)
    scores = np.einsum('ihd,ijhd->ihj', Q_mh, K_mh) * scale

    weights = softmax_np(scores, axis=-1)

    attn_out = np.einsum('ihj,ijhd->ihd', weights, V_mh)
    attn_out = attn_out.reshape(N, D)

    return attn_out @ W_o


def transformer_block_np(node_features, frac_coords, lattice,
                         W_q, W_k, W_v, W_ek, W_ev, W_o,
                         W_ffn1, b_ffn1, W_ffn2, b_ffn2,
                         gate_attn, gate_ffn,
                         num_heads=4):
    """NumPy reference for TransformerBlock."""
    N, D = node_features.shape
    nh = num_heads
    hd = D // nh

    # ---- Attention sub-layer ----
    pos_i = np.expand_dims(frac_coords, 1)
    pos_j = np.expand_dims(frac_coords, 0)
    frac_diff = (pos_j - pos_i) % 1.0

    lattice_ip = lattice @ lattice.T
    lattice_ip_flat = lattice_ip.ravel()

    ip_edges = np.broadcast_to(
        lattice_ip_flat.reshape(1, 1, 9), (N, N, 9))
    edge_feats = np.concatenate([ip_edges, frac_diff], axis=-1)

    Q = node_features @ W_q
    K_n = node_features @ W_k
    V_n = node_features @ W_v

    K_e = np.einsum('ija,da->ijd', edge_feats, W_ek)
    V_e = np.einsum('ija,da->ijd', edge_feats, W_ev)

    K_t = np.expand_dims(K_n, 0) + K_e
    V_t = np.expand_dims(V_n, 0) + V_e

    Q_mh = Q.reshape(N, nh, hd)
    K_mh = K_t.reshape(N, N, nh, hd)
    V_mh = V_t.reshape(N, N, nh, hd)

    scale = 1.0 / np.sqrt(hd)
    scores = np.einsum('ihd,ijhd->ihj', Q_mh, K_mh) * scale
    weights = softmax_np(scores, axis=-1)

    attn_out = np.einsum('ihj,ijhd->ihd', weights, V_mh)
    attn_out = attn_out.reshape(N, D)
    attn_out = attn_out @ W_o

    # Gated residual #1
    x = node_features + gate_attn * attn_out

    # ---- FFN sub-layer ----
    ffn_hidden = x @ W_ffn1 + b_ffn1
    ffn_act = np.maximum(ffn_hidden, 0.0)  # ReLU
    ffn_out = ffn_act @ W_ffn2 + b_ffn2

    return x + gate_ffn * ffn_out


# ---------------------------------------------------------------------------
# Tests (torch tensors on NPU)
# ---------------------------------------------------------------------------

def _random_weights(D, seed=42):
    rng = np.random.RandomState(seed)
    return {
        'W_q': rng.randn(D, D).astype(np.float32) * 0.02,
        'W_k': rng.randn(D, D).astype(np.float32) * 0.02,
        'W_v': rng.randn(D, D).astype(np.float32) * 0.02,
        'W_o': rng.randn(D, D).astype(np.float32) * 0.02,
        'W_ek': rng.randn(D, 12).astype(np.float32) * 0.02,
        'W_ev': rng.randn(D, 12).astype(np.float32) * 0.02,
    }


def test_crysformer_attention():
    np.random.seed(42)
    D = 64
    nh = 4
    N = 8

    node_feats = np.random.randn(N, D).astype(np.float32)
    frac_coords = np.random.rand(N, 3).astype(np.float32)
    lattice = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float32)

    w = _random_weights(D, seed=0)

    # NPU kernel
    nf_t = torch.from_numpy(node_feats).to('npu')
    fc_t = torch.from_numpy(frac_coords).to('npu')
    lat_t = torch.from_numpy(lattice).to('npu')
    w_q_t = torch.from_numpy(w['W_q']).to('npu')
    w_k_t = torch.from_numpy(w['W_k']).to('npu')
    w_v_t = torch.from_numpy(w['W_v']).to('npu')
    w_ek_t = torch.from_numpy(w['W_ek']).to('npu')
    w_ev_t = torch.from_numpy(w['W_ev']).to('npu')
    w_o_t = torch.from_numpy(w['W_o']).to('npu')
    out_t = torch.zeros(N, D, dtype=torch.float32, device='npu')

    CrysFormerAttention(
        nf_t, fc_t, lat_t,
        w_q_t, w_k_t, w_v_t, w_ek_t, w_ev_t, w_o_t,
        out_t,
        torch.tensor(D, device='npu'),
        torch.tensor(nh, device='npu'),
    )

    result = out_t.cpu().numpy()

    # NumPy reference
    expected = crysformer_attention_np(
        node_feats, frac_coords, lattice,
        w['W_q'], w['W_k'], w['W_v'], w['W_ek'], w['W_ev'], w['W_o'],
        num_heads=nh,
    )

    err = np.abs(result - expected).max()
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  CrysFormerAttention: {status}")


def test_transformer_block():
    np.random.seed(42)
    D = 64
    nh = 4
    N = 8

    node_feats = np.random.randn(N, D).astype(np.float32)
    frac_coords = np.random.rand(N, 3).astype(np.float32)
    lattice = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float32)

    w = _random_weights(D, seed=0)
    w_ffn1 = np.random.randn(D, D * 4).astype(np.float32) * 0.02
    b_ffn1 = np.zeros(D * 4, dtype=np.float32)
    w_ffn2 = np.random.randn(D * 4, D).astype(np.float32) * 0.02
    b_ffn2 = np.zeros(D, dtype=np.float32)
    gate_a = np.ones(D, dtype=np.float32) * 0.5
    gate_f = np.ones(D, dtype=np.float32) * 0.5

    nf_t = torch.from_numpy(node_feats).to('npu')
    fc_t = torch.from_numpy(frac_coords).to('npu')
    lat_t = torch.from_numpy(lattice).to('npu')

    tensors = {k: torch.from_numpy(v).to('npu')
               for k, v in {**w,
                            'W_ffn1': w_ffn1, 'b_ffn1': b_ffn1,
                            'W_ffn2': w_ffn2, 'b_ffn2': b_ffn2,
                            'gate_attn': gate_a, 'gate_ffn': gate_f}.items()}

    out_t = torch.zeros(N, D, dtype=torch.float32, device='npu')
    TransformerBlock(
        nf_t, fc_t, lat_t,
        tensors['W_q'], tensors['W_k'], tensors['W_v'],
        tensors['W_ek'], tensors['W_ev'], tensors['W_o'],
        tensors['W_ffn1'], tensors['b_ffn1'],
        tensors['W_ffn2'], tensors['b_ffn2'],
        tensors['gate_attn'], tensors['gate_ffn'],
        out_t,
        torch.tensor(D, device='npu'),
        torch.tensor(nh, device='npu'),
    )

    result = out_t.cpu().numpy()

    expected = transformer_block_np(
        node_feats, frac_coords, lattice,
        w['W_q'], w['W_k'], w['W_v'], w['W_ek'], w['W_ev'], w['W_o'],
        w_ffn1, b_ffn1, w_ffn2, b_ffn2, gate_a, gate_f,
        num_heads=nh,
    )

    err = np.abs(result - expected).max()
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  TransformerBlock:   {status}")


def main():
    print("=" * 50)
    print("PyPTO Attention Ops — Tests")
    print("=" * 50)
    test_crysformer_attention()
    test_transformer_block()
    print("=" * 50)


if __name__ == "__main__":
    main()
