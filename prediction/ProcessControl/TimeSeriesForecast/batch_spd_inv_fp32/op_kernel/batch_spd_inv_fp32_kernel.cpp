#include "kernel_operator.h"

using namespace AscendC;

namespace {

// fp32 batched symmetric-positive-definite matrix inverse.
//
//   gi[b] = inv(g[b])        g[b] is m x m, SPD (Gram matrix x[b] @ x[b]^T)
//
// This is the missing primitive behind Koopa's DMD step. Koopa solves the
// underdetermined system  x @ K = y  with torch.linalg.lstsq, whose min-norm
// closed form is  K = x^T (x x^T)^-1 y. On the NPU the two big matmuls run
// natively (torch.bmm, ~0.04ms), but the tiny m x m Gram inverse silently
// falls back to CPU (torch.linalg.inv / lstsq unsupported), costing
// 1260-4289ms per call. This op moves that inverse on-device.
//
// m is small (Koopa: 3-7; cap 64). The inverse is computed with an
// LDL^T factorization (no square roots -> only +,-,*,/ on the scalar unit):
//   g = L D L^T  with L unit-lower-triangular, D diagonal
//   g^-1 = L^-T D^-1 L^-1
// L is inverted by forward substitution (unit triangular). All arithmetic is
// scalar on the Vector core; batches are distributed across cores. No Cube,
// no Matmul API, no cross-core sync.

struct BatchSpdInvTiling {
  uint32_t batch;
  uint32_t m;
};

class BatchSpdInvFp32Kernel {
 public:
  __aicore__ inline BatchSpdInvFp32Kernel() = default;

  TPipe pipe;

  __aicore__ inline void Init(GM_ADDR g, GM_ADDR gi,
                              uint32_t batch, uint32_t m) {
    batch_ = batch;
    m_ = m;
    mm_ = m * m;
    mmUp_ = (mm_ + 7) / 8 * 8;  // round up to 32B (8 floats) for UB alignment
    gGm_.SetGlobalBuffer((__gm__ float*)g, batch_ * mm_);
    giGm_.SetGlobalBuffer((__gm__ float*)gi, batch_ * mm_);
    // Per-batch g/gi blocks are DMA'd into UB; all scalar work happens in UB.
    // Direct scalar GetValue/SetValue on global memory is NOT cache-coherent
    // across AIV cores, so we never touch GM with the scalar unit.
    pipe.InitBuffer(gBuf_, mmUp_ * sizeof(float));
    pipe.InitBuffer(giBuf_, mmUp_ * sizeof(float));
    pipe.InitBuffer(lBuf_, mm_ * sizeof(float));
    pipe.InitBuffer(invBuf_, mm_ * sizeof(float));
    pipe.InitBuffer(dBuf_, m_ * sizeof(float));
    pipe.InitBuffer(diBuf_, m_ * sizeof(float));
  }

  __aicore__ inline void Process() {
    LocalTensor<float> gL = gBuf_.Get<float>();
    LocalTensor<float> giL = giBuf_.Get<float>();
    LocalTensor<float> L = lBuf_.Get<float>();
    LocalTensor<float> Minv = invBuf_.Get<float>();
    LocalTensor<float> d = dBuf_.Get<float>();
    LocalTensor<float> di = diBuf_.Get<float>();

    const uint32_t blk = GetBlockIdx();
    const uint32_t nblk = GetBlockNum();
    const uint32_t M = m_;

    DataCopyExtParams copyParams{1, mm_ * (uint32_t)sizeof(float), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    for (uint32_t b = blk; b < batch_; b += nblk) {
      const uint64_t base = (uint64_t)b * mm_;

      // ---- DMA g[b] from GM into UB ----
      DataCopyPad(gL, gGm_[base], copyParams, padParams);
      pipe_barrier(PIPE_ALL);

      // ---- pivot floor for numerical robustness ----
      // A real Gram x[b] x[b]^T can be rank-deficient (collinear / constant
      // channels, short context), making an LDL pivot d[j] tiny or non-positive
      // in fp32. The unguarded 1/d[j] then yields Inf/NaN that silently poisons
      // gi[b] and every downstream bmm. We clamp each pivot to a small floor
      // relative to the matrix scale (= max|diag g|). This is a localized ridge:
      // it is bit-identical on well-conditioned SPD inputs (the Koopa norm) and
      // keeps the output finite & bounded on degenerate inputs instead of NaN.
      // It is a stability guard, not lstsq's min-norm pseudo-inverse.
      float scale = 0.0f;
      for (uint32_t j = 0; j < M; ++j) {
        float gjj = gL.GetValue(j * M + j);
        float a = gjj >= 0.0f ? gjj : -gjj;
        if (a > scale) scale = a;
      }
      if (scale <= 0.0f) scale = 1.0f;
      const float pivotFloor = 1e-6f * scale;

      // ---- LDL^T factorization of g[b] (read g from UB) ----
      for (uint32_t j = 0; j < M; ++j) {
        float dj = gL.GetValue(j * M + j);
        for (uint32_t k = 0; k < j; ++k) {
          float ljk = L.GetValue(j * M + k);
          dj -= ljk * ljk * d.GetValue(k);
        }
        if (dj < pivotFloor) dj = pivotFloor;
        d.SetValue(j, dj);
        float invdj = 1.0f / dj;
        L.SetValue(j * M + j, 1.0f);
        for (uint32_t i = j + 1; i < M; ++i) {
          float s = gL.GetValue(i * M + j);
          for (uint32_t k = 0; k < j; ++k) {
            s -= L.GetValue(i * M + k) * d.GetValue(k) * L.GetValue(j * M + k);
          }
          L.SetValue(i * M + j, s * invdj);
        }
      }
      for (uint32_t k = 0; k < M; ++k) {
        di.SetValue(k, 1.0f / d.GetValue(k));
      }

      // ---- Minv = L^-1 (unit lower triangular, forward substitution) ----
      for (uint32_t j = 0; j < M; ++j) {
        Minv.SetValue(j * M + j, 1.0f);
        for (uint32_t i = j + 1; i < M; ++i) {
          float s = 0.0f;
          for (uint32_t k = j; k < i; ++k) {
            s -= L.GetValue(i * M + k) * Minv.GetValue(k * M + j);
          }
          Minv.SetValue(i * M + j, s);
        }
      }

      // ---- gi = Minv^T D^-1 Minv (symmetric) ----
      // gi[i][j] = sum_{k=max(i,j)}^{M-1} Minv[k][i] * di[k] * Minv[k][j]
      for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = i; j < M; ++j) {
          float s = 0.0f;
          uint32_t kstart = j;  // j >= i here
          for (uint32_t k = kstart; k < M; ++k) {
            s += Minv.GetValue(k * M + i) * di.GetValue(k) * Minv.GetValue(k * M + j);
          }
          giL.SetValue(i * M + j, s);
          if (j != i) {
            giL.SetValue(j * M + i, s);
          }
        }
      }

      // ---- DMA gi[b] from UB back to GM ----
      pipe_barrier(PIPE_ALL);
      DataCopyPad(giGm_[base], giL, copyParams);
      pipe_barrier(PIPE_ALL);
    }
  }

 private:
  TBuf<TPosition::VECCALC> gBuf_;
  TBuf<TPosition::VECCALC> giBuf_;
  TBuf<TPosition::VECCALC> lBuf_;
  TBuf<TPosition::VECCALC> invBuf_;
  TBuf<TPosition::VECCALC> dBuf_;
  TBuf<TPosition::VECCALC> diBuf_;

  GlobalTensor<float> gGm_;
  GlobalTensor<float> giGm_;
  uint32_t batch_ = 0;
  uint32_t m_ = 0;
  uint32_t mm_ = 0;
  uint32_t mmUp_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void batch_spd_inv_fp32(
    GM_ADDR g, GM_ADDR gi, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  const BatchSpdInvTiling* tp =
      reinterpret_cast<const BatchSpdInvTiling*>(&tilingData);
  BatchSpdInvFp32Kernel op;
  op.Init(g, gi, tp->batch, tp->m);
  op.Process();
}
