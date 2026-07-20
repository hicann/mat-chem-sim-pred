#include "kernel_operator.h"

using namespace AscendC;

namespace {

// CfC (Closed-form Continuous-time, Hasani et al. 2022) fused sequence scan, fp32.
//
// Fuses the launch-bound per-timestep CfC recurrence into a single kernel launch.
// Per batch b and timestep s (state h in R^H, input x_s in R^IN):
//
//   z    = concat(h, x_s)                         (len H+IN)
//   ff1  = tanh   (Wf1 z + bf1)                    candidate A  (len H)
//   ff2  = tanh   (Wf2 z + bf2)                    candidate B  (len H)
//   gate = sigmoid(Wt  z + bt )                    time-interp  (len H)
//   h    = ff1 * (1 - gate) + ff2 * gate
//   out[b, s, :] = h
//
// h feeds back through tanh/sigmoid each step, so there is no parallel/chunked
// form. Batches are independent: each AI core owns a contiguous slab of batches
// and runs its whole time loop locally, holding h in UB -- no SyncAll, no GM
// state round-trip, no workspace.
//
// `weight` is the transposed, column-packed projection [H+IN, 3H]:
//   weight[:, 0:H]   = Wf1.T   (candidate A on z)
//   weight[:, H:2H]  = Wf2.T   (candidate B on z)
//   weight[:, 2H:3H] = Wt.T    (time-interp gate on z)
// so each projection is an axpy accumulation over the H+IN contraction axis,
// matching the transposed-weight convention of the GRU / TiRex / LEM kernels.
// `bias` is [3H] = concat(bf1, bf2, bt).

struct CfcScanFusedTiling {
  uint32_t batch;    // B
  uint32_t length;   // L
  uint32_t in_size;  // IN
  uint32_t hidden;   // H
};

class CfcScanFusedKernel {
 public:
  __aicore__ inline CfcScanFusedKernel() = default;

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                              GM_ADDR output, const CfcScanFusedTiling* t) {
    B_ = t->batch;
    L_ = t->length;
    IN_ = t->in_size;
    H_ = t->hidden;
    din_ = H_ + IN_;
    w3_ = 3U * H_;

    xGm_.SetGlobalBuffer((__gm__ float*)x, (uint64_t)B_ * L_ * IN_);
    wGm_.SetGlobalBuffer((__gm__ float*)weight, (uint64_t)din_ * w3_);
    bGm_.SetGlobalBuffer((__gm__ float*)bias, w3_);
    outGm_.SetGlobalBuffer((__gm__ float*)output, (uint64_t)B_ * L_ * H_);

    const uint32_t ldMax = (din_ > w3_) ? din_ : w3_;
    pipe_.InitBuffer(ldQue_, 2, ldMax * sizeof(float));
    pipe_.InitBuffer(stQue_, 2, H_ * sizeof(float));

    pipe_.InitBuffer(wBuf_, (uint64_t)din_ * w3_ * sizeof(float));
    pipe_.InitBuffer(biasBuf_, w3_ * sizeof(float));
    pipe_.InitBuffer(preBuf_, w3_ * sizeof(float));
    pipe_.InitBuffer(wtmpBuf_, w3_ * sizeof(float));
    pipe_.InitBuffer(sgTmpBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(zBuf_, din_ * sizeof(float));
    pipe_.InitBuffer(uBuf_, IN_ * sizeof(float));
    pipe_.InitBuffer(hBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(ff1Buf_, H_ * sizeof(float));
    pipe_.InitBuffer(ff2Buf_, H_ * sizeof(float));
    pipe_.InitBuffer(gBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(omBuf_, H_ * sizeof(float));
  }

  __aicore__ inline void Process() {
    uint32_t blockNum = GetBlockNum();
    if (blockNum == 0U) blockNum = 1U;
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t base = B_ / blockNum;
    const uint32_t tail = B_ % blockNum;
    const uint32_t count = base + (blockIdx < tail ? 1U : 0U);
    const uint32_t start = blockIdx * base + (blockIdx < tail ? blockIdx : tail);

    // Persistent UB residents: full weight [din,3H] and bias[3H].
    LocalTensor<float> wfull = wBuf_.Get<float>();
    for (uint32_t c = 0; c < din_; ++c) {
      LocalTensor<float> q = ldQue_.AllocTensor<float>();
      DataCopy(q, wGm_[(uint64_t)c * w3_], w3_);
      ldQue_.EnQue(q);
      q = ldQue_.DeQue<float>();
      Adds(wfull[c * w3_], q, 0.0f, w3_);
      ldQue_.FreeTensor(q);
    }
    LocalTensor<float> bb = biasBuf_.Get<float>();
    {
      LocalTensor<float> q = ldQue_.AllocTensor<float>();
      DataCopy(q, bGm_, w3_);
      ldQue_.EnQue(q);
      q = ldQue_.DeQue<float>();
      Adds(bb, q, 0.0f, w3_);
      ldQue_.FreeTensor(q);
    }

    for (uint32_t i = 0; i < count; ++i) {
      ProcessBatch(start + i, wfull, bb);
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

  // tanh(x) = 2*sigmoid(2x) - 1; dst may alias src; tmp distinct, len n.
  __aicore__ inline void Tanh(const LocalTensor<float>& dst,
                              const LocalTensor<float>& src,
                              const LocalTensor<float>& tmp, uint32_t n) {
    Muls(dst, src, 2.0f, n);
    Sigmoid(dst, dst, tmp, n);
    Muls(dst, dst, 2.0f, n);
    Adds(dst, dst, -1.0f, n);
  }

  // pre[0:n] = sum_c z[c] * W[c, col0 + 0:n]  (no bias here)
  __aicore__ inline void Project(const LocalTensor<float>& pre,
                                 const LocalTensor<float>& z,
                                 const LocalTensor<float>& wfull,
                                 const LocalTensor<float>& wtmp,
                                 uint32_t col0, uint32_t n) {
    Duplicate(pre, 0.0f, n);
    for (uint32_t c = 0; c < din_; ++c) {
      const float zc = z.GetValue(c);
      Muls(wtmp, wfull[c * w3_ + col0], zc, n);
      Add(pre, pre, wtmp, n);
    }
  }

  __aicore__ inline void ProcessBatch(uint32_t b,
                                      const LocalTensor<float>& wfull,
                                      const LocalTensor<float>& bb) {
    LocalTensor<float> h = hBuf_.Get<float>();
    Duplicate(h, 0.0f, H_);

    LocalTensor<float> z = zBuf_.Get<float>();
    LocalTensor<float> ub = uBuf_.Get<float>();
    LocalTensor<float> pre = preBuf_.Get<float>();
    LocalTensor<float> wtmp = wtmpBuf_.Get<float>();
    LocalTensor<float> sg = sgTmpBuf_.Get<float>();
    LocalTensor<float> ff1 = ff1Buf_.Get<float>();
    LocalTensor<float> ff2 = ff2Buf_.Get<float>();
    LocalTensor<float> g = gBuf_.Get<float>();
    LocalTensor<float> om = omBuf_.Get<float>();

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

      // z = concat(h, x_s); project all 3 heads at once over cols[0:3H]
      Adds(z[0], h, 0.0f, H_);
      Adds(z[H_], ub, 0.0f, IN_);
      Project(pre, z, wfull, wtmp, 0U, w3_);
      Add(pre, pre, bb, w3_);

      Tanh(ff1, pre[0], sg, H_);        // ff1  = tanh(cand A)
      Tanh(ff2, pre[H_], sg, H_);       // ff2  = tanh(cand B)
      Sigmoid(g, pre[2U * H_], sg, H_); // gate = sigmoid(time-interp)

      // h = ff1*(1-gate) + ff2*gate
      Muls(om, g, -1.0f, H_);
      Adds(om, om, 1.0f, H_);           // om = 1 - gate
      Mul(ff1, ff1, om, H_);            // ff1*(1-gate)
      Mul(ff2, ff2, g, H_);             // ff2*gate
      Add(h, ff1, ff2, H_);             // h = ff1*(1-gate) + ff2*gate

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
  TBuf<TPosition::VECCALC> biasBuf_;
  TBuf<TPosition::VECCALC> preBuf_;
  TBuf<TPosition::VECCALC> wtmpBuf_;
  TBuf<TPosition::VECCALC> sgTmpBuf_;
  TBuf<TPosition::VECCALC> zBuf_;
  TBuf<TPosition::VECCALC> uBuf_;
  TBuf<TPosition::VECCALC> hBuf_;
  TBuf<TPosition::VECCALC> ff1Buf_;
  TBuf<TPosition::VECCALC> ff2Buf_;
  TBuf<TPosition::VECCALC> gBuf_;
  TBuf<TPosition::VECCALC> omBuf_;

  GlobalTensor<float> xGm_;
  GlobalTensor<float> wGm_;
  GlobalTensor<float> bGm_;
  GlobalTensor<float> outGm_;
  uint32_t B_ = 0;
  uint32_t L_ = 0;
  uint32_t IN_ = 0;
  uint32_t H_ = 0;
  uint32_t din_ = 0;
  uint32_t w3_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void cfc_scan_fused(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling) {
  (void)workspace;
  GET_TILING_DATA(tilingData, tiling);
  CfcScanFusedKernel op;
  op.Init(x, weight, bias, output,
          reinterpret_cast<const CfcScanFusedTiling*>(&tilingData));
  op.Process();
}
