#include "kernel_operator.h"

using namespace AscendC;

namespace {

// LTC (Liquid Time-Constant network, Hasani et al. AAAI'21) fused ODE-solver scan, fp32.
//
// Fuses the launch-bound per-timestep LTC fused semi-implicit ODE solver into a single
// kernel launch. Per batch b, timestep s (state h in R^H, input x_s in R^IN), with K
// inner solver unfolds:
//
//   wx = W_in x_s                            (constant across unfolds)
//   repeat K times:
//     pre = W_rec h + wx + b
//     f   = sigmoid(pre)
//     h   = (cm/dt * h + gleak*Eleak + f*A) / (cm/dt + gleak + f)
//   out[b, s, :] = h
//
// Each unfold recomputes the hidden-hidden matmul W_rec h -> K matmuls + elementwise
// division per timestep. Nonlinear gated recurrence with an inner ODE solver -> no
// parallel/chunked form; NPU has no native LTC primitive. Batches are independent: each
// AI core owns a contiguous slab of batches and runs its whole time loop locally,
// holding h in UB -- no SyncAll, no GM state round-trip, no workspace.
//
// `weight` is the column-packed transposed projection [H+IN, H]:
//   weight[0:H, :]   = W_rec.T   (hidden-hidden, matmul over H)
//   weight[H:H+IN,:] = W_in.T    (input projection, matmul over IN)
// `bias` is [5H] packed per-neuron: [ b | gleak | Eleak | cm | A ].
// dt and K are baked constants (match the framework baseline).

constexpr float DT = 0.042f;
constexpr uint32_t K_UNFOLD = 6U;

struct LtcScanFusedTiling {
  uint32_t batch;    // B
  uint32_t length;   // L
  uint32_t in_size;  // IN
  uint32_t hidden;   // H
};

class LtcScanFusedKernel {
 public:
  __aicore__ inline LtcScanFusedKernel() = default;

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                              GM_ADDR output, const LtcScanFusedTiling* t) {
    B_ = t->batch;
    L_ = t->length;
    IN_ = t->in_size;
    H_ = t->hidden;
    din_ = H_ + IN_;

    xGm_.SetGlobalBuffer((__gm__ float*)x, (uint64_t)B_ * L_ * IN_);
    wGm_.SetGlobalBuffer((__gm__ float*)weight, (uint64_t)din_ * H_);
    bGm_.SetGlobalBuffer((__gm__ float*)bias, (uint64_t)5U * H_);
    outGm_.SetGlobalBuffer((__gm__ float*)output, (uint64_t)B_ * L_ * H_);

    const uint32_t ldMax = (IN_ > H_) ? IN_ : H_;
    pipe_.InitBuffer(ldQue_, 2, ldMax * sizeof(float));
    pipe_.InitBuffer(stQue_, 2, H_ * sizeof(float));

    pipe_.InitBuffer(wBuf_, (uint64_t)din_ * H_ * sizeof(float));
    pipe_.InitBuffer(bias5Buf_, (uint64_t)5U * H_ * sizeof(float));
    pipe_.InitBuffer(cmdtBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(gEBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(dencBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(wxBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(preBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(fBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(numBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(denBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(wtmpBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(sgTmpBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(uBuf_, IN_ * sizeof(float));
    pipe_.InitBuffer(hBuf_, H_ * sizeof(float));
  }

  __aicore__ inline void Process() {
    uint32_t blockNum = GetBlockNum();
    if (blockNum == 0U) blockNum = 1U;
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t base = B_ / blockNum;
    const uint32_t tail = B_ % blockNum;
    const uint32_t count = base + (blockIdx < tail ? 1U : 0U);
    const uint32_t start = blockIdx * base + (blockIdx < tail ? blockIdx : tail);

    // Persistent UB residents: full weight [H+IN,H] and packed bias[5H].
    LocalTensor<float> wfull = wBuf_.Get<float>();
    for (uint32_t c = 0; c < din_; ++c) {
      LocalTensor<float> q = ldQue_.AllocTensor<float>();
      DataCopy(q, wGm_[(uint64_t)c * H_], H_);
      ldQue_.EnQue(q);
      q = ldQue_.DeQue<float>();
      Adds(wfull[c * H_], q, 0.0f, H_);
      ldQue_.FreeTensor(q);
    }
    LocalTensor<float> b5 = bias5Buf_.Get<float>();
    for (uint32_t c = 0; c < 5U; ++c) {
      LocalTensor<float> q = ldQue_.AllocTensor<float>();
      DataCopy(q, bGm_[(uint64_t)c * H_], H_);
      ldQue_.EnQue(q);
      q = ldQue_.DeQue<float>();
      Adds(b5[c * H_], q, 0.0f, H_);
      ldQue_.FreeTensor(q);
    }

    LocalTensor<float> bb = b5;                         // b      = bias[0:H]
    LocalTensor<float> gleak = b5[(uint64_t)1U * H_];   // gleak  = bias[H:2H]
    LocalTensor<float> Eleak = b5[(uint64_t)2U * H_];   // Eleak  = bias[2H:3H]
    LocalTensor<float> cm = b5[(uint64_t)3U * H_];      // cm     = bias[3H:4H]
    LocalTensor<float> Acond = b5[(uint64_t)4U * H_];   // A      = bias[4H:5H]

    // precompute time-constant terms
    LocalTensor<float> cmdt = cmdtBuf_.Get<float>();
    Muls(cmdt, cm, 1.0f / DT, H_);                      // cm/dt
    LocalTensor<float> gE = gEBuf_.Get<float>();
    Mul(gE, gleak, Eleak, H_);                          // gleak*Eleak
    LocalTensor<float> denc = dencBuf_.Get<float>();
    Add(denc, cmdt, gleak, H_);                         // cm/dt + gleak

    for (uint32_t i = 0; i < count; ++i) {
      ProcessBatch(start + i, wfull, bb, Acond, cmdt, gE, denc);
    }
  }

 private:
  // sigmoid(x) = 1/(1+exp(-x)); dst may alias src; tmp must be distinct, len n.
  __aicore__ inline void Sigmoid(const LocalTensor<float>& dst,
                                 const LocalTensor<float>& src,
                                 const LocalTensor<float>& tmp, uint32_t n) {
    Muls(tmp, src, -1.0f, n);
    Mins(tmp, tmp, 80.0f, n);
    Maxs(tmp, tmp, -80.0f, n);
    Exp(tmp, tmp, n);
    Adds(tmp, tmp, 1.0f, n);
    Duplicate(dst, 1.0f, n);
    Div(dst, dst, tmp, n);
  }

  // pre[0:H] = sum_c src[c] * W[c, 0:H]  over c in [0, nContract)
  __aicore__ inline void Project(const LocalTensor<float>& pre,
                                 const LocalTensor<float>& src,
                                 const LocalTensor<float>& wfull,
                                 const LocalTensor<float>& wtmp,
                                 uint32_t nContract) {
    Duplicate(pre, 0.0f, H_);
    for (uint32_t c = 0; c < nContract; ++c) {
      const float sc = src.GetValue(c);
      Muls(wtmp, wfull[c * H_], sc, H_);
      Add(pre, pre, wtmp, H_);
    }
  }

  __aicore__ inline void ProcessBatch(uint32_t b,
                                      const LocalTensor<float>& wfull,
                                      const LocalTensor<float>& bb,
                                      const LocalTensor<float>& Acond,
                                      const LocalTensor<float>& cmdt,
                                      const LocalTensor<float>& gE,
                                      const LocalTensor<float>& denc) {
    LocalTensor<float> h = hBuf_.Get<float>();
    Duplicate(h, 0.0f, H_);

    LocalTensor<float> ub = uBuf_.Get<float>();
    LocalTensor<float> wx = wxBuf_.Get<float>();
    LocalTensor<float> pre = preBuf_.Get<float>();
    LocalTensor<float> f = fBuf_.Get<float>();
    LocalTensor<float> num = numBuf_.Get<float>();
    LocalTensor<float> den = denBuf_.Get<float>();
    LocalTensor<float> wtmp = wtmpBuf_.Get<float>();
    LocalTensor<float> sg = sgTmpBuf_.Get<float>();

    for (uint32_t s = 0; s < L_; ++s) {
      // load x_s [IN] (length need not be 32B-aligned -> DataCopyPad)
      {
        LocalTensor<float> q = ldQue_.AllocTensor<float>();
        DataCopyExtParams cp{1U, (uint32_t)(IN_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        DataCopyPad(q, xGm_[((uint64_t)b * L_ + s) * IN_], cp, pad);
        ldQue_.EnQue(q);
        q = ldQue_.DeQue<float>();
        Adds(ub, q, 0.0f, IN_);
        ldQue_.FreeTensor(q);
      }
      // wx = W_in x_s  (W_in rows start at H in wfull); constant across unfolds
      Project(wx, ub, wfull[(uint64_t)H_ * H_], wtmp, IN_);

      for (uint32_t k = 0; k < K_UNFOLD; ++k) {
        // pre = W_rec h + wx + b
        Project(pre, h, wfull, wtmp, H_);
        Add(pre, pre, wx, H_);
        Add(pre, pre, bb, H_);
        // f = sigmoid(pre)
        Sigmoid(f, pre, sg, H_);
        // num = cm/dt * h + gleak*Eleak + f*A
        Mul(num, cmdt, h, H_);
        Add(num, num, gE, H_);
        Mul(wtmp, f, Acond, H_);
        Add(num, num, wtmp, H_);
        // den = (cm/dt + gleak) + f
        Add(den, denc, f, H_);
        // h = num/den
        Div(h, num, den, H_);
      }

      // store out[b, s, :] = h
      LocalTensor<float> o = stQue_.AllocTensor<float>();
      Adds(o, h, 0.0f, H_);
      stQue_.EnQue(o);
      o = stQue_.DeQue<float>();
      DataCopyExtParams cpo{1U, (uint32_t)(H_ * sizeof(float)), 0U, 0U, 0U};
      DataCopyPad(outGm_[((uint64_t)b * L_ + s) * H_], o, cpo);
      stQue_.FreeTensor(o);
    }
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, 2> ldQue_;
  TQue<TPosition::VECOUT, 2> stQue_;
  TBuf<TPosition::VECCALC> wBuf_;
  TBuf<TPosition::VECCALC> bias5Buf_;
  TBuf<TPosition::VECCALC> cmdtBuf_;
  TBuf<TPosition::VECCALC> gEBuf_;
  TBuf<TPosition::VECCALC> dencBuf_;
  TBuf<TPosition::VECCALC> wxBuf_;
  TBuf<TPosition::VECCALC> preBuf_;
  TBuf<TPosition::VECCALC> fBuf_;
  TBuf<TPosition::VECCALC> numBuf_;
  TBuf<TPosition::VECCALC> denBuf_;
  TBuf<TPosition::VECCALC> wtmpBuf_;
  TBuf<TPosition::VECCALC> sgTmpBuf_;
  TBuf<TPosition::VECCALC> uBuf_;
  TBuf<TPosition::VECCALC> hBuf_;

  GlobalTensor<float> xGm_;
  GlobalTensor<float> wGm_;
  GlobalTensor<float> bGm_;
  GlobalTensor<float> outGm_;
  uint32_t B_ = 0;
  uint32_t L_ = 0;
  uint32_t IN_ = 0;
  uint32_t H_ = 0;
  uint32_t din_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void ltc_scan_fused(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling) {
  (void)workspace;
  GET_TILING_DATA(tilingData, tiling);
  LtcScanFusedKernel op;
  op.Init(x, weight, bias, output,
          reinterpret_cast<const LtcScanFusedTiling*>(&tilingData));
  op.Process();
}
