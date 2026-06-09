#ifndef DEEPONET_KERNEL_H
#define DEEPONET_KERNEL_H

#include "kernel_operator.h"
#include "pde_math.h"

using namespace AscendC;

struct DeepOnetTilingData {
    int32_t batchSize;
    int32_t branchDim;
    int32_t trunkDim;
    int32_t latentDim;
    int32_t querySize;
    int32_t tileSize;
    int32_t coreNum;
    int32_t branchWeightSize;
    int32_t trunkWeightSize;
};

constexpr int32_t DON_MAX_LATENT = 256;
constexpr int32_t DON_MAX_QUERY = 1024;
constexpr int32_t DON_BLOCK_SIZE = 32;
constexpr int32_t DON_MAX_CORES = 32;

__aicore__ inline uint32_t DonAlignUp(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

class DeepOnetOp {
public:
    __aicore__ inline DeepOnetOp() {}

    __aicore__ inline void Init(GM_ADDR branchInput, GM_ADDR trunkInput,
                                 GM_ADDR branchWeights, GM_ADDR trunkWeights,
                                 GM_ADDR output, GM_ADDR tiling) {
        __gm__ const DeepOnetTilingData* td =
            reinterpret_cast<__gm__ const DeepOnetTilingData*>(tiling);

        batchSize_ = td->batchSize;
        branchDim_ = td->branchDim;
        trunkDim_ = td->trunkDim;
        latentDim_ = td->latentDim;
        querySize_ = td->querySize;
        tileSize_ = td->tileSize;
        coreNum_ = td->coreNum;

        coreIdx_ = GetBlockIdx();

        int32_t workPerCore = (batchSize_ + coreNum_ - 1) / coreNum_;
        myStart_ = coreIdx_ * workPerCore;
        myEnd_ = myStart_ + workPerCore;
        if (myEnd_ > batchSize_) myEnd_ = batchSize_;
        if (myStart_ >= batchSize_) {
            myStart_ = 0;
            myEnd_ = 0;
        }
        myCount_ = myEnd_ - myStart_;

        branchInput_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(branchInput));
        trunkInput_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(trunkInput));
        branchWeights_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(branchWeights));
        trunkWeights_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(trunkWeights));
        output_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    }

    __aicore__ inline void FCPredict(const float* input, float* output,
                                      const GlobalTensor<float>& weights,
                                      int32_t inDim, int32_t outDim,
                                      int32_t weightOffset) {
        for (int32_t j = 0; j < outDim; j++) {
            float sum = 0.0f;
            int32_t wBase = weightOffset + j * inDim;
            for (int32_t k = 0; k < inDim; k++) {
                sum += input[k] * weights.GetValue(wBase + k);
            }
            float bias = weights.GetValue(weightOffset + inDim * outDim + j);
            output[j] = pde_math::pde_tanhf(sum + bias);
        }
    }

    __aicore__ inline void Process() {
        if (myCount_ <= 0) return;

        float branchCode[DON_MAX_LATENT];
        float trunkCode[DON_MAX_LATENT];

        for (int32_t b = 0; b < myCount_; b++) {
            int32_t batchIdx = myStart_ + b;

            float branchIn[DON_MAX_LATENT];
            for (int32_t d = 0; d < branchDim_; d++) {
                branchIn[d] = branchInput_gm_.GetValue(batchIdx * branchDim_ + d);
            }

            FCPredict(branchIn, branchCode, branchWeights_gm_,
                       branchDim_, latentDim_, 0);

            for (int32_t q = 0; q < querySize_; q++) {
                float trunkIn[DON_MAX_LATENT];
                for (int32_t d = 0; d < trunkDim_; d++) {
                    trunkIn[d] = trunkInput_gm_.GetValue(q * trunkDim_ + d);
                }

                FCPredict(trunkIn, trunkCode, trunkWeights_gm_,
                           trunkDim_, latentDim_, 0);

                float result = 0.0f;
                for (int32_t k = 0; k < latentDim_; k++) {
                    result += branchCode[k] * trunkCode[k];
                }

                output_gm_.SetValue((batchIdx * querySize_ + q), result);
            }
        }
    }

private:
    GlobalTensor<float> branchInput_gm_;
    GlobalTensor<float> trunkInput_gm_;
    GlobalTensor<float> branchWeights_gm_;
    GlobalTensor<float> trunkWeights_gm_;
    GlobalTensor<float> output_gm_;

    TPipe pipe_;

    int32_t batchSize_;
    int32_t branchDim_;
    int32_t trunkDim_;
    int32_t latentDim_;
    int32_t querySize_;
    int32_t tileSize_;
    int32_t coreNum_;
    int32_t coreIdx_;
    int32_t myStart_;
    int32_t myEnd_;
    int32_t myCount_;
};

#endif
