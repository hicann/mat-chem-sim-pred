#include "kernel_operator.h"

using namespace AscendC;

namespace {

// UnICORNN (undamped independent controlled oscillatory RNN, Rusch & Mishra ICML'21)
// fused sequence scan, fp32.
//
// Fuses the launch-bound per-timestep UnICORNN recurrence into a single kernel
// launch. Per batch b, timestep s (states y,z in R^H, input x_s in R^IN):
//
//   arg   = w (.) y + V x_s + b           (w is DIAGONAL recurrent: elementwise)
//   sig_c = sigmoid(c)                     (per-neuron control step size, time-const)
//   z     = z - dt*sig_c*( tanh(arg) + alpha*y )
//   y     = y + dt*sig_c*z
//   out[b, s, :] = y
//
// Hidden-hidden coupling is diagonal (independent per channel); only the input
// projection V x_s is a matmul. Nonlinear tanh feedback -> no parallel/chunked form;
// NPU has no native UnICORNN primitive. Batches are independent: each AI core owns a
// contiguous slab of batches and runs its whole time loop locally, holding y,z in
// UB -- no SyncAll, no GM state round-trip, no workspace.
//
// `weight` is the column-packed transposed projection [IN+2, H]:
//   weight[0:IN, :] = V.T   (input projection, matmul over IN)
//   weight[IN,   :] = w     (diagonal recurrent weight, elementwise)
//   weight[IN+1, :] = c     (per-neuron control)
// `bias` is [H] = b. dt, alpha are baked constants (match the framework baseline).

constexpr float DT = 0.042f;
constexpr float ALPHA = 1.0f;

struct UnicornnScanFusedTiling {
  uint32_t batch;    // B
  uint32_t length;   // L
  uint32_t in_size;  // IN
  uint32_t hidden;   // H
};

class UnicornnScanFusedKernel {
 public:
  __aicore__ inline UnicornnScanFusedKernel() = default;

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                              GM_ADDR output, const UnicornnScanFusedTiling* t) {
    B_ = t->batch;
    L_ = t->length;
    IN_ = t->in_size;
    H_ = t->hidden;
    wrows_ = IN_ + 2U;

    xGm_.SetGlobalBuffer((__gm__ float*)x, (uint64_t)B_ * L_ * IN_);
    wGm_.SetGlobalBuffer((__gm__ float*)weight, (uint64_t)wrows_ * H_);
    bGm_.SetGlobalBuffer((__gm__ float*)bias, H_);
    outGm_.SetGlobalBuffer((__gm__ float*)output, (uint64_t)B_ * L_ * H_);

    const uint32_t ldMax = (IN_ > H_) ? IN_ : H_;
    pipe_.InitBuffer(ldQue_, 2, ldMax * sizeof(float));
    pipe_.InitBuffer(stQue_, 2, H_ * sizeof(float));

    pipe_.InitBuffer(wBuf_, (uint64_t)wrows_ * H_ * sizeof(float));
    pipe_.InitBuffer(biasBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(sigcBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(preBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(argBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(wtmpBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(sgTmpBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(uBuf_, IN_ * sizeof(float));
    pipe_.InitBuffer(yBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(velBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(tBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(accBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(wkBuf_, H_ * sizeof(float));
  }

  __aicore__ inline void Process() {
    uint32_t blockNum = GetBlockNum();
    if (blockNum == 0U) blockNum = 1U;
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t base = B_ / blockNum;
    const uint32_t tail = B_ % blockNum;
    const uint32_t count = base + (blockIdx < tail ? 1U : 0U);
    const uint32_t start = blockIdx * base + (blockIdx < tail ? blockIdx : tail);

    // Persistent UB residents: full weight [IN+2,H] and bias[H].
    LocalTensor<float> wfull = wBuf_.Get<float>();
    for (uint32_t c = 0; c < wrows_; ++c) {
      LocalTensor<float> q = ldQue_.AllocTensor<float>();
      DataCopy(q, wGm_[(uint64_t)c * H_], H_);
      ldQue_.EnQue(q);
      q = ldQue_.DeQue<float>();
      Adds(wfull[c * H_], q, 0.0f, H_);
      ldQue_.FreeTensor(q);
    }
    LocalTensor<float> bb = biasBuf_.Get<float>();
    {
      LocalTensor<float> q = ldQue_.AllocTensor<float>();
      DataCopy(q, bGm_, H_);
      ldQue_.EnQue(q);
      q = ldQue_.DeQue<float>();
      Adds(bb, q, 0.0f, H_);
      ldQue_.FreeTensor(q);
    }

    // sig_c = sigmoid(c) is time-constant; compute once. c = wfull row IN+1.
    LocalTensor<float> sigc = sigcBuf_.Get<float>();
    LocalTensor<float> sg = sgTmpBuf_.Get<float>();
    Sigmoid(sigc, wfull[(uint64_t)(IN_ + 1U) * H_], sg, H_);

    LocalTensor<float> wdiag = wfull[(uint64_t)IN_ * H_];  // diagonal recurrent w

    for (uint32_t i = 0; i < count; ++i) {
      ProcessBatch(start + i, wfull, bb, sigc, wdiag, sg);
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

  // pre[0:H] = sum_c src[c] * W[c, 0:H]  over c in [0, nContract)  (no bias)
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
                                      const LocalTensor<float>& sigc,
                                      const LocalTensor<float>& wdiag,
                                      const LocalTensor<float>& sg) {
    LocalTensor<float> y = yBuf_.Get<float>();
    LocalTensor<float> vel = velBuf_.Get<float>();
    Duplicate(y, 0.0f, H_);
    Duplicate(vel, 0.0f, H_);

    LocalTensor<float> ub = uBuf_.Get<float>();
    LocalTensor<float> pre = preBuf_.Get<float>();
    LocalTensor<float> arg = argBuf_.Get<float>();
    LocalTensor<float> wtmp = wtmpBuf_.Get<float>();
    LocalTensor<float> t = tBuf_.Get<float>();
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> wk = wkBuf_.Get<float>();

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

      // arg = w (.) y + V x_s + b
      Project(pre, ub, wfull, wtmp, IN_);   // pre = V x_s
      Mul(arg, wdiag, y, H_);               // arg = w (.) y
      Add(arg, arg, pre, H_);               // arg += V x_s
      Add(arg, arg, bb, H_);                // arg += b

      Tanh(t, arg, sg, H_);                 // t = tanh(arg)

      // z = z - dt*sig_c*( tanh(arg) + alpha*y )
      Muls(acc, y, ALPHA, H_);              // acc = alpha*y
      Add(acc, acc, t, H_);                 // acc = tanh(arg) + alpha*y
      Mul(acc, acc, sigc, H_);              // acc = sig_c*(...)
      Muls(acc, acc, DT, H_);               // acc = dt*sig_c*(...)
      Sub(vel, vel, acc, H_);               // z = z - acc

      // y = y + dt*sig_c*z
      Mul(wk, sigc, vel, H_);               // wk = sig_c*z
      Muls(wk, wk, DT, H_);                 // wk = dt*sig_c*z
      Add(y, y, wk, H_);                    // y = y + wk

      // store out[b, s, :] = y
      LocalTensor<float> o = stQue_.AllocTensor<float>();
      Adds(o, y, 0.0f, H_);
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
  TBuf<TPosition::VECCALC> sigcBuf_;
  TBuf<TPosition::VECCALC> preBuf_;
  TBuf<TPosition::VECCALC> argBuf_;
  TBuf<TPosition::VECCALC> wtmpBuf_;
  TBuf<TPosition::VECCALC> sgTmpBuf_;
  TBuf<TPosition::VECCALC> uBuf_;
  TBuf<TPosition::VECCALC> yBuf_;
  TBuf<TPosition::VECCALC> velBuf_;
  TBuf<TPosition::VECCALC> tBuf_;
  TBuf<TPosition::VECCALC> accBuf_;
  TBuf<TPosition::VECCALC> wkBuf_;

  GlobalTensor<float> xGm_;
  GlobalTensor<float> wGm_;
  GlobalTensor<float> bGm_;
  GlobalTensor<float> outGm_;
  uint32_t B_ = 0;
  uint32_t L_ = 0;
  uint32_t IN_ = 0;
  uint32_t H_ = 0;
  uint32_t wrows_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void unicornn_scan_fused(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling) {
  (void)workspace;
  GET_TILING_DATA(tilingData, tiling);
  UnicornnScanFusedKernel op;
  op.Init(x, weight, bias, output,
          reinterpret_cast<const UnicornnScanFusedTiling*>(&tilingData));
  op.Process();
}
