#include "kernel_operator.h"

using namespace AscendC;

namespace {

// coRNN (coupled oscillatory RNN, Rusch & Mishra, ICLR'21) fused sequence scan, fp32.
//
// Fuses the launch-bound per-timestep coRNN IMEX recurrence into a single kernel
// launch. Per batch b and timestep s (states y,z in R^H, input x_s in R^IN):
//
//   pre   = Wy y + Wz z + V x_s + b                (len H)
//   z_new = z + dt*( tanh(pre) - gamma*y - eps*z )
//   y_new = y + dt*z_new
//   out[b, s, :] = y_new
//
// Two coupled states (y position, z velocity) coupled through tanh every step, so
// there is no parallel/chunked form. Batches are independent: each AI core owns a
// contiguous slab of batches and runs its whole time loop locally, holding y,z in
// UB -- no SyncAll, no GM state round-trip, no workspace.
//
// `weight` is the transposed, column-packed projection [2H+IN, H]:
//   weight[0:H,     :] = Wy.T   (on y)
//   weight[H:2H,    :] = Wz.T   (on z)
//   weight[2H:2H+IN,:] = V.T    (on x_s)
// so the projection is an axpy accumulation over the 2H+IN contraction axis,
// matching the transposed-weight convention of the GRU / TiRex / LEM / CfC kernels.
// `bias` is [H]. dt, gamma, eps are baked constants (match the framework baseline).

constexpr float DT = 0.042f;
constexpr float GAMMA = 1.0f;
constexpr float EPS = 1.0f;

struct CornnScanFusedTiling {
  uint32_t batch;    // B
  uint32_t length;   // L
  uint32_t in_size;  // IN
  uint32_t hidden;   // H
};

class CornnScanFusedKernel {
 public:
  __aicore__ inline CornnScanFusedKernel() = default;

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                              GM_ADDR output, const CornnScanFusedTiling* t) {
    B_ = t->batch;
    L_ = t->length;
    IN_ = t->in_size;
    H_ = t->hidden;
    din_ = 2U * H_ + IN_;

    xGm_.SetGlobalBuffer((__gm__ float*)x, (uint64_t)B_ * L_ * IN_);
    wGm_.SetGlobalBuffer((__gm__ float*)weight, (uint64_t)din_ * H_);
    bGm_.SetGlobalBuffer((__gm__ float*)bias, H_);
    outGm_.SetGlobalBuffer((__gm__ float*)output, (uint64_t)B_ * L_ * H_);

    const uint32_t ldMax = (din_ > H_) ? din_ : H_;
    pipe_.InitBuffer(ldQue_, 2, ldMax * sizeof(float));
    pipe_.InitBuffer(stQue_, 2, H_ * sizeof(float));

    pipe_.InitBuffer(wBuf_, (uint64_t)din_ * H_ * sizeof(float));
    pipe_.InitBuffer(biasBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(preBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(wtmpBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(sgTmpBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(zcatBuf_, din_ * sizeof(float));
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

    // Persistent UB residents: full weight [din,H] and bias[H].
    LocalTensor<float> wfull = wBuf_.Get<float>();
    for (uint32_t c = 0; c < din_; ++c) {
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

  // pre[0:H] = sum_c zcat[c] * W[c, 0:H]  (no bias here)
  __aicore__ inline void Project(const LocalTensor<float>& pre,
                                 const LocalTensor<float>& zcat,
                                 const LocalTensor<float>& wfull,
                                 const LocalTensor<float>& wtmp) {
    Duplicate(pre, 0.0f, H_);
    for (uint32_t c = 0; c < din_; ++c) {
      const float zc = zcat.GetValue(c);
      Muls(wtmp, wfull[c * H_], zc, H_);
      Add(pre, pre, wtmp, H_);
    }
  }

  __aicore__ inline void ProcessBatch(uint32_t b,
                                      const LocalTensor<float>& wfull,
                                      const LocalTensor<float>& bb) {
    LocalTensor<float> y = yBuf_.Get<float>();
    LocalTensor<float> vel = velBuf_.Get<float>();
    Duplicate(y, 0.0f, H_);
    Duplicate(vel, 0.0f, H_);

    LocalTensor<float> zcat = zcatBuf_.Get<float>();
    LocalTensor<float> ub = uBuf_.Get<float>();
    LocalTensor<float> pre = preBuf_.Get<float>();
    LocalTensor<float> wtmp = wtmpBuf_.Get<float>();
    LocalTensor<float> sg = sgTmpBuf_.Get<float>();
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

      // zcat = concat(y, z, x_s); pre = zcat @ W + b
      Adds(zcat[0], y, 0.0f, H_);
      Adds(zcat[H_], vel, 0.0f, H_);
      Adds(zcat[2U * H_], ub, 0.0f, IN_);
      Project(pre, zcat, wfull, wtmp);
      Add(pre, pre, bb, H_);

      Tanh(t, pre, sg, H_);             // t = tanh(pre)

      // acc = tanh(pre) - gamma*y - eps*z
      Muls(acc, y, -GAMMA, H_);         // acc = -gamma*y
      Add(acc, acc, t, H_);             // acc = t - gamma*y
      Muls(wk, vel, -EPS, H_);          // wk  = -eps*z
      Add(acc, acc, wk, H_);            // acc = t - gamma*y - eps*z

      // z = z + dt*acc
      Muls(acc, acc, DT, H_);
      Add(vel, vel, acc, H_);

      // y = y + dt*z
      Muls(wk, vel, DT, H_);
      Add(y, y, wk, H_);

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
  TBuf<TPosition::VECCALC> preBuf_;
  TBuf<TPosition::VECCALC> wtmpBuf_;
  TBuf<TPosition::VECCALC> sgTmpBuf_;
  TBuf<TPosition::VECCALC> zcatBuf_;
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
  uint32_t din_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void cornn_scan_fused(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling) {
  (void)workspace;
  GET_TILING_DATA(tilingData, tiling);
  CornnScanFusedKernel op;
  op.Init(x, weight, bias, output,
          reinterpret_cast<const CornnScanFusedTiling*>(&tilingData));
  op.Process();
}
