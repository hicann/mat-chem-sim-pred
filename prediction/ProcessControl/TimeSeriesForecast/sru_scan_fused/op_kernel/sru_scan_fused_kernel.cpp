#include "kernel_operator.h"

using namespace AscendC;

namespace {

// SRU (Simple Recurrent Unit, Lei & Zhang 2018)
// fused sequence scan, fp32.
//
// Per batch b, timestep s:
//   x_tilde = W x_s                              // input projection
//   f = sigmoid(Wf x_s + v_f * c + b_f)          // forget gate with peephole on c
//   r = sigmoid(Wr x_s + v_r * c + b_r)          // reset gate with peephole on c
//   c = f * c + (1-f) * x_tilde                   // cell update
//   h = r * tanh(c) + (1-r) * x_tilde             // highway output
//   out[b, s, :] = h
//
// weight [3IN, H] column-packed transposed:
//   [0:IN, :]      = W^T    (input projection)
//   [IN:2IN, :]    = Wf^T   (forget gate input)
//   [2IN:3IN, :]   = Wr^T   (reset gate input)
// bias [4H]: [v_f | v_r | b_f | b_r]

struct SruScanFusedTiling {
  uint32_t batch;
  uint32_t length;
  uint32_t in_size;
  uint32_t hidden;
};

class SruScanFusedKernel {
 public:
  __aicore__ inline SruScanFusedKernel() = default;

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                              GM_ADDR output, const SruScanFusedTiling* t) {
    B_ = t->batch; L_ = t->length; IN_ = t->in_size; H_ = t->hidden;
    din_ = 3U * IN_;

    xGm_.SetGlobalBuffer((__gm__ float*)x, (uint64_t)B_ * L_ * IN_);
    wGm_.SetGlobalBuffer((__gm__ float*)weight, (uint64_t)din_ * H_);
    bGm_.SetGlobalBuffer((__gm__ float*)bias, 4U * H_);
    outGm_.SetGlobalBuffer((__gm__ float*)output, (uint64_t)B_ * L_ * H_);

    const uint32_t ldMax = (IN_ > H_) ? IN_ : H_;
    pipe_.InitBuffer(ldQue_, 2, ldMax * sizeof(float));
    pipe_.InitBuffer(stQue_, 2, H_ * sizeof(float));
    pipe_.InitBuffer(wBuf_, (uint64_t)din_ * H_ * sizeof(float));
    pipe_.InitBuffer(vfBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(vrBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(bfBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(brBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(xtBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(fBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(rBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(projBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(wtmpBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(uBuf_, IN_ * sizeof(float));
    pipe_.InitBuffer(cBuf_, H_ * sizeof(float));
    pipe_.InitBuffer(onesBuf_, H_ * sizeof(float));
  }

  __aicore__ inline void Process() {
    uint32_t blockNum = GetBlockNum();
    if (blockNum == 0U) blockNum = 1U;
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t base = B_ / blockNum;
    const uint32_t tail = B_ % blockNum;
    const uint32_t count = base + (blockIdx < tail ? 1U : 0U);
    const uint32_t start = blockIdx * base + (blockIdx < tail ? blockIdx : tail);

    LocalTensor<float> wfull = wBuf_.Get<float>();
    for (uint32_t c = 0; c < din_; ++c) {
      LocalTensor<float> q = ldQue_.AllocTensor<float>();
      DataCopy(q, wGm_[(uint64_t)c * H_], H_);
      ldQue_.EnQue(q); q = ldQue_.DeQue<float>();
      Adds(wfull[c * H_], q, 0.0f, H_);
      ldQue_.FreeTensor(q);
    }

    LocalTensor<float> vf = vfBuf_.Get<float>();
    LocalTensor<float> vr = vrBuf_.Get<float>();
    LocalTensor<float> bff = bfBuf_.Get<float>();
    LocalTensor<float> br = brBuf_.Get<float>();
    auto loadBias = [&](LocalTensor<float>& dst, uint32_t off) {
      LocalTensor<float> q = ldQue_.AllocTensor<float>();
      DataCopy(q, bGm_[off], H_); ldQue_.EnQue(q);
      q = ldQue_.DeQue<float>(); Adds(dst, q, 0.0f, H_); ldQue_.FreeTensor(q);
    };
    loadBias(vf, 0); loadBias(vr, H_); loadBias(bff, 2U*H_); loadBias(br, 3U*H_);

    for (uint32_t i = 0; i < count; ++i) {
      ProcessBatch(start + i, wfull, vf, vr, bff, br);
    }
  }

 private:
  __aicore__ inline void Project(const LocalTensor<float>& dst,
                                 const LocalTensor<float>& src,
                                 const LocalTensor<float>& wfull,
                                 const LocalTensor<float>& wtmp,
                                 uint32_t offRow, uint32_t nRows) {
    Duplicate(dst, 0.0f, H_);
    for (uint32_t c = 0; c < nRows; ++c) {
      const float sc = src.GetValue(c);
      Muls(wtmp, wfull[(offRow + c) * H_], sc, H_);
      Add(dst, dst, wtmp, H_);
    }
  }

  __aicore__ inline void Sigmoid(const LocalTensor<float>& dst,
                                  const LocalTensor<float>& src) {
    LocalTensor<float> ones = onesBuf_.Get<float>();
    Duplicate(ones, 1.0f, H_);
    Muls(dst, src, -1.0f, H_);
    Maxs(dst, dst, -20.0f, H_);
    Mins(dst, dst, 20.0f, H_);
    Exp(dst, dst, H_);
    Adds(dst, dst, 1.0f, H_);
    Div(dst, ones, dst, H_);
  }

  __aicore__ inline void Tanh(const LocalTensor<float>& dst,
                               const LocalTensor<float>& src) {
    LocalTensor<float> denom = onesBuf_.Get<float>();
    Muls(dst, src, 2.0f, H_);
    Maxs(dst, dst, -20.0f, H_);
    Mins(dst, dst, 20.0f, H_);
    Exp(dst, dst, H_);
    Adds(denom, dst, 1.0f, H_);
    Adds(dst, dst, -1.0f, H_);
    Div(dst, dst, denom, H_);
  }

  __aicore__ inline void ProcessBatch(uint32_t b,
                                      const LocalTensor<float>& wfull,
                                      const LocalTensor<float>& vf,
                                      const LocalTensor<float>& vr,
                                      const LocalTensor<float>& bff,
                                      const LocalTensor<float>& br) {
    LocalTensor<float> cell = cBuf_.Get<float>();
    Duplicate(cell, 0.0f, H_);

    LocalTensor<float> ub = uBuf_.Get<float>();
    LocalTensor<float> xt = xtBuf_.Get<float>();
    LocalTensor<float> f = fBuf_.Get<float>();
    LocalTensor<float> r = rBuf_.Get<float>();
    LocalTensor<float> proj = projBuf_.Get<float>();
    LocalTensor<float> wtmp = wtmpBuf_.Get<float>();

    for (uint32_t s = 0; s < L_; ++s) {
      {
        LocalTensor<float> q = ldQue_.AllocTensor<float>();
        DataCopyExtParams cp{1U, (uint32_t)(IN_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        DataCopyPad(q, xGm_[((uint64_t)b * L_ + s) * IN_], cp, pad);
        ldQue_.EnQue(q); q = ldQue_.DeQue<float>();
        Adds(ub, q, 0.0f, IN_); ldQue_.FreeTensor(q);
      }

      // x_tilde = W x_s
      Project(xt, ub, wfull, wtmp, 0, IN_);

      // f = sigmoid(Wf x_s + v_f * c + b_f)
      Project(f, ub, wfull, wtmp, IN_, IN_);
      Mul(proj, vf, cell, H_);
      Add(f, f, proj, H_);
      Add(f, f, bff, H_);
      Sigmoid(f, f);

      // r = sigmoid(Wr x_s + v_r * c + b_r)
      Project(r, ub, wfull, wtmp, 2U * IN_, IN_);
      Mul(proj, vr, cell, H_);
      Add(r, r, proj, H_);
      Add(r, r, br, H_);
      Sigmoid(r, r);

      // c = f * c + (1-f) * x_tilde
      Mul(cell, f, cell, H_);
      Muls(wtmp, f, -1.0f, H_);
      Adds(wtmp, wtmp, 1.0f, H_);
      Mul(wtmp, wtmp, xt, H_);
      Add(cell, cell, wtmp, H_);

      // h = r * tanh(c) + (1-r) * x_tilde
      Tanh(proj, cell);
      Mul(proj, r, proj, H_);
      Muls(wtmp, r, -1.0f, H_);
      Adds(wtmp, wtmp, 1.0f, H_);
      Mul(wtmp, wtmp, xt, H_);
      Add(proj, proj, wtmp, H_);

      // Store h
      LocalTensor<float> o = stQue_.AllocTensor<float>();
      Adds(o, proj, 0.0f, H_);
      stQue_.EnQue(o); o = stQue_.DeQue<float>();
      DataCopyExtParams cpo{1U, (uint32_t)(H_ * sizeof(float)), 0U, 0U, 0U};
      DataCopyPad(outGm_[((uint64_t)b * L_ + s) * H_], o, cpo);
      stQue_.FreeTensor(o);
    }
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, 2> ldQue_;
  TQue<TPosition::VECOUT, 2> stQue_;
  TBuf<TPosition::VECCALC> wBuf_, vfBuf_, vrBuf_, bfBuf_, brBuf_;
  TBuf<TPosition::VECCALC> xtBuf_, fBuf_, rBuf_, projBuf_, wtmpBuf_, uBuf_, cBuf_, onesBuf_;

  GlobalTensor<float> xGm_, wGm_, bGm_, outGm_;
  uint32_t B_ = 0, L_ = 0, IN_ = 0, H_ = 0, din_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void sru_scan_fused(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling) {
  (void)workspace;
  GET_TILING_DATA(tilingData, tiling);
  SruScanFusedKernel op;
  op.Init(x, weight, bias, output,
          reinterpret_cast<const SruScanFusedTiling*>(&tilingData));
  op.Process();
}
