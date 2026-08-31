/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "../../geometry_math.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
class DimeNetTripletAngleFusedKernel
{
   public:
    __aicore__ inline DimeNetTripletAngleFusedKernel() = default;

    __aicore__ inline void Init(GM_ADDR position, GM_ADDR idxI, GM_ADDR idxJ, GM_ADDR idxK, GM_ADDR angle,
                                uint32_t nodes, uint32_t triplets)
    {
        nodes_ = nodes;
        triplets_ = triplets;
        positionGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(position), static_cast<uint64_t>(nodes_) * 3U);
        idxIGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxI), triplets_);
        idxJGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxJ), triplets_);
        idxKGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxK), triplets_);
        angleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(angle), triplets_);
        pipe_.InitBuffer(rootInputBuf_, 8U * sizeof(float));
        pipe_.InitBuffer(rootOutputBuf_, 8U * sizeof(float));
        pipe_.InitBuffer(outputBuf_, 8U * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> rootInput = rootInputBuf_.Get<float>();
        LocalTensor<float> rootOutput = rootOutputBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        const DataCopyExtParams copy{1U, sizeof(float), 0U, 0U, 0U};
        for (uint32_t index = GetBlockIdx(); index < triplets_; index += GetBlockNum())
        {
            const int32_t atomI = idxIGm_.GetValue(index);
            const int32_t atomJ = idxJGm_.GetValue(index);
            const int32_t atomK = idxKGm_.GetValue(index);
            if (!ValidNode(atomI) || !ValidNode(atomJ) || !ValidNode(atomK))
            {
                Duplicate(output, 0.0F, 8U);
                DataCopyPad(angleGm_[index], output, copy);
                continue;
            }
            float i[3], j[3], k[3];
            LoadPosition(atomI, i);
            LoadPosition(atomJ, j);
            LoadPosition(atomK, k);
            float ij[3], jk[3];
            for (uint32_t lane = 0U; lane < 3U; ++lane)
            {
                ij[lane] = i[lane] - j[lane];
                jk[lane] = j[lane] - k[lane];
            }
            const float dot = MolecularGeometry::Dot(ij, jk);
            const float crossSquare = MolecularGeometry::CrossSquare(ij, jk);
            output.SetValue(0U, MolecularGeometry::AngleFromValues(dot, crossSquare, rootInput, rootOutput));
            pipe_barrier(PIPE_ALL);
            DataCopyPad(angleGm_[index], output, copy);
            pipe_barrier(PIPE_ALL);
        }
    }

   private:
    __aicore__ inline bool ValidNode(int32_t node) const { return node >= 0 && static_cast<uint32_t>(node) < nodes_; }

    __aicore__ inline void LoadPosition(int32_t node, float value[3])
    {
        const uint64_t base = static_cast<uint64_t>(node) * 3U;
        value[0] = positionGm_.GetValue(base);
        value[1] = positionGm_.GetValue(base + 1U);
        value[2] = positionGm_.GetValue(base + 2U);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rootInputBuf_, rootOutputBuf_, outputBuf_;
    GlobalTensor<float> positionGm_, angleGm_;
    GlobalTensor<int32_t> idxIGm_, idxJGm_, idxKGm_;
    uint32_t nodes_ = 0U;
    uint32_t triplets_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void dimenet_triplet_angle_fused_kernel(GM_ADDR position, GM_ADDR idxI, GM_ADDR idxJ,
                                                                         GM_ADDR idxK, GM_ADDR angle, uint32_t nodes,
                                                                         uint32_t triplets)
{
    DimeNetTripletAngleFusedKernel op;
    op.Init(position, idxI, idxJ, idxK, angle, nodes, triplets);
    op.Process();
}
