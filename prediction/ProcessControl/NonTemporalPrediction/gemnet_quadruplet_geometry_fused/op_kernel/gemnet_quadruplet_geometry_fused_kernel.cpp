/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "../../geometry_math.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace
{
class GemNetQuadrupletGeometryFusedKernel
{
   public:
    __aicore__ inline GemNetQuadrupletGeometryFusedKernel() = default;
    __aicore__ inline void Init(GM_ADDR position, GM_ADDR sourceIndex, GM_ADDR targetIndex, GM_ADDR interactionSource,
                                GM_ADDR interactionTarget, GM_ADDR reduceCa, GM_ADDR expandDb,
                                GM_ADDR reduceIntermediateCa, GM_ADDR expandIntermediateDb,
                                GM_ADDR reduceIntermediateAb, GM_ADDR expandIntermediateAb, GM_ADDR angleCab,
                                GM_ADDR angleAbd, GM_ADDR thetaCabd, uint32_t nodes, uint32_t edges,
                                uint32_t interactionEdges, uint32_t quadruplets, uint32_t intermediateCa,
                                uint32_t intermediateDb)
    {
        nodes_ = nodes;
        edges_ = edges;
        interactionEdges_ = interactionEdges;
        quadruplets_ = quadruplets;
        intermediateCa_ = intermediateCa;
        intermediateDb_ = intermediateDb;
        positionGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(position), static_cast<uint64_t>(nodes_) * 3U);
        sourceIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndex), edges_);
        targetIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(targetIndex), edges_);
        interactionSourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(interactionSource), interactionEdges_);
        interactionTargetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(interactionTarget), interactionEdges_);
        reduceCaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(reduceCa), quadruplets_);
        expandDbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expandDb), quadruplets_);
        reduceIntermediateCaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(reduceIntermediateCa),
                                                intermediateCa_);
        expandIntermediateDbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expandIntermediateDb),
                                                intermediateDb_);
        reduceIntermediateAbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(reduceIntermediateAb),
                                                intermediateCa_);
        expandIntermediateAbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expandIntermediateAb),
                                                intermediateDb_);
        angleCabGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(angleCab), intermediateCa_);
        angleAbdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(angleAbd), intermediateDb_);
        thetaCabdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(thetaCabd), quadruplets_);
        pipe_.InitBuffer(rootInputBuf_, 8U * sizeof(float));
        pipe_.InitBuffer(rootOutputBuf_, 8U * sizeof(float));
        pipe_.InitBuffer(outputBuf_, 8U * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        ProcessIntermediateCa();
        ProcessIntermediateDb();
        ProcessQuadruplets();
    }

   private:
    __aicore__ inline void ProcessIntermediateCa()
    {
        LocalTensor<float> rootInput = rootInputBuf_.Get<float>();
        LocalTensor<float> rootOutput = rootOutputBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        const DataCopyExtParams copy{1U, sizeof(float), 0U, 0U, 0U};
        for (uint32_t index = GetBlockIdx(); index < intermediateCa_; index += GetBlockNum())
        {
            const int32_t edgeCa = reduceIntermediateCaGm_.GetValue(index);
            const int32_t interaction = reduceIntermediateAbGm_.GetValue(index);
            if (!ValidEdge(edgeCa) || !ValidInteraction(interaction))
            {
                Store(angleCabGm_[index], 0.0F, output, copy);
                continue;
            }
            const int32_t atomC = sourceIndexGm_.GetValue(edgeCa);
            const int32_t atomA = targetIndexGm_.GetValue(edgeCa);
            const int32_t atomB = interactionSourceGm_.GetValue(interaction);
            Store(angleCabGm_[index], Angle(atomB, atomA, atomC, rootInput, rootOutput), output, copy);
        }
    }

    __aicore__ inline void ProcessIntermediateDb()
    {
        LocalTensor<float> rootInput = rootInputBuf_.Get<float>();
        LocalTensor<float> rootOutput = rootOutputBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        const DataCopyExtParams copy{1U, sizeof(float), 0U, 0U, 0U};
        for (uint32_t index = GetBlockIdx(); index < intermediateDb_; index += GetBlockNum())
        {
            const int32_t edgeDb = expandIntermediateDbGm_.GetValue(index);
            const int32_t interaction = expandIntermediateAbGm_.GetValue(index);
            if (!ValidEdge(edgeDb) || !ValidInteraction(interaction))
            {
                Store(angleAbdGm_[index], 0.0F, output, copy);
                continue;
            }
            const int32_t atomD = sourceIndexGm_.GetValue(edgeDb);
            const int32_t atomB = targetIndexGm_.GetValue(edgeDb);
            const int32_t atomA = interactionTargetGm_.GetValue(interaction);
            Store(angleAbdGm_[index], Angle(atomA, atomB, atomD, rootInput, rootOutput), output, copy);
        }
    }

    __aicore__ inline void ProcessQuadruplets()
    {
        LocalTensor<float> rootInput = rootInputBuf_.Get<float>();
        LocalTensor<float> rootOutput = rootOutputBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        const DataCopyExtParams copy{1U, sizeof(float), 0U, 0U, 0U};
        for (uint32_t index = GetBlockIdx(); index < quadruplets_; index += GetBlockNum())
        {
            ProcessQuadruplet(index, rootInput, rootOutput, output, copy);
        }
    }

    __aicore__ inline void ProcessQuadruplet(uint32_t index, LocalTensor<float>& rootInput,
                                             LocalTensor<float>& rootOutput, LocalTensor<float>& output,
                                             const DataCopyExtParams& copy)
    {
        const int32_t edgeCa = reduceCaGm_.GetValue(index);
        const int32_t edgeDb = expandDbGm_.GetValue(index);
        if (!ValidEdge(edgeCa) || !ValidEdge(edgeDb))
        {
            Store(thetaCabdGm_[index], 0.0F, output, copy);
            return;
        }
        const int32_t atomC = sourceIndexGm_.GetValue(edgeCa);
        const int32_t atomA = targetIndexGm_.GetValue(edgeCa);
        const int32_t atomD = sourceIndexGm_.GetValue(edgeDb);
        const int32_t atomB = targetIndexGm_.GetValue(edgeDb);
        float a[3], b[3], c[3], d[3];
        LoadPosition(atomA, a);
        LoadPosition(atomB, b);
        LoadPosition(atomC, c);
        LoadPosition(atomD, d);
        float rac[3], rab[3], rba[3], rbd[3];
        for (uint32_t lane = 0U; lane < 3U; ++lane)
        {
            rac[lane] = c[lane] - a[lane];
            rab[lane] = b[lane] - a[lane];
            rba[lane] = a[lane] - b[lane];
            rbd[lane] = d[lane] - b[lane];
        }
        const float rabSquare = Dot(rab, rab);
        const float rbaSquare = Dot(rba, rba);
        const float racScale = ProjectionScale(MolecularGeometry::Dot(rac, rab), rabSquare);
        const float rbdScale = ProjectionScale(MolecularGeometry::Dot(rbd, rba), rbaSquare);
        float projectedAc[3], projectedBd[3];
        for (uint32_t lane = 0U; lane < 3U; ++lane)
        {
            projectedAc[lane] = rac[lane] - racScale * rab[lane];
            projectedBd[lane] = rbd[lane] - rbdScale * rba[lane];
        }
        const float value = MolecularGeometry::AngleFromValues(MolecularGeometry::Dot(projectedAc, projectedBd),
                                                               MolecularGeometry::CrossSquare(projectedAc, projectedBd),
                                                               rootInput, rootOutput);
        Store(thetaCabdGm_[index], value, output, copy);
    }

    __aicore__ inline void Store(GlobalTensor<float> destination, float value, LocalTensor<float>& output,
                                 const DataCopyExtParams& copy)
    {
        Duplicate(output, 0.0F, 8U);
        pipe_barrier(PIPE_ALL);
        output.SetValue(0U, value);
        pipe_barrier(PIPE_ALL);
        DataCopyPad(destination, output, copy);
        pipe_barrier(PIPE_ALL);
    }
    __aicore__ inline float ProjectionScale(float numerator, float denominator) const
    {
        const float safeDenominator = denominator > MolecularGeometry::kTiny ? denominator : 1.0F;
        return denominator > MolecularGeometry::kTiny ? numerator / safeDenominator : 0.0F;
    }
    __aicore__ inline bool ValidEdge(int32_t edge) const { return edge >= 0 && static_cast<uint32_t>(edge) < edges_; }
    __aicore__ inline bool ValidInteraction(int32_t edge) const
    {
        return edge >= 0 && static_cast<uint32_t>(edge) < interactionEdges_;
    }
    __aicore__ inline void LoadPosition(int32_t node, float value[3])
    {
        const uint64_t base = static_cast<uint64_t>(node) * 3U;
        value[0] = positionGm_.GetValue(base);
        value[1] = positionGm_.GetValue(base + 1U);
        value[2] = positionGm_.GetValue(base + 2U);
    }
    __aicore__ inline float Angle(int32_t first, int32_t middle, int32_t second, LocalTensor<float>& rootInput,
                                  LocalTensor<float>& rootOutput)
    {
        float firstPosition[3], middlePosition[3], secondPosition[3];
        LoadPosition(first, firstPosition);
        LoadPosition(middle, middlePosition);
        LoadPosition(second, secondPosition);
        float left[3], right[3];
        for (uint32_t lane = 0U; lane < 3U; ++lane)
        {
            left[lane] = firstPosition[lane] - middlePosition[lane];
            right[lane] = secondPosition[lane] - middlePosition[lane];
        }
        return MolecularGeometry::AngleFromValues(Dot(left, right), CrossSquare(left, right), rootInput, rootOutput);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rootInputBuf_, rootOutputBuf_, outputBuf_;
    GlobalTensor<float> positionGm_, angleCabGm_, angleAbdGm_, thetaCabdGm_;
    GlobalTensor<int32_t> sourceIndexGm_, targetIndexGm_, interactionSourceGm_, interactionTargetGm_;
    GlobalTensor<int32_t> reduceCaGm_, expandDbGm_, reduceIntermediateCaGm_, expandIntermediateDbGm_,
        reduceIntermediateAbGm_, expandIntermediateAbGm_;
    uint32_t nodes_ = 0U, edges_ = 0U, interactionEdges_ = 0U, quadruplets_ = 0U, intermediateCa_ = 0U,
             intermediateDb_ = 0U;
};
}  // namespace

extern "C" __global__ __aicore__ void gemnet_quadruplet_geometry_fused_kernel(
    GM_ADDR position, GM_ADDR sourceIndex, GM_ADDR targetIndex, GM_ADDR interactionSource, GM_ADDR interactionTarget,
    GM_ADDR reduceCa, GM_ADDR expandDb, GM_ADDR reduceIntermediateCa, GM_ADDR expandIntermediateDb,
    GM_ADDR reduceIntermediateAb, GM_ADDR expandIntermediateAb, GM_ADDR angleCab, GM_ADDR angleAbd, GM_ADDR thetaCabd,
    uint32_t nodes, uint32_t edges, uint32_t interactionEdges, uint32_t quadruplets, uint32_t intermediateCa,
    uint32_t intermediateDb)
{
    GemNetQuadrupletGeometryFusedKernel op;
    op.Init(position, sourceIndex, targetIndex, interactionSource, interactionTarget, reduceCa, expandDb,
            reduceIntermediateCa, expandIntermediateDb, reduceIntermediateAb, expandIntermediateAb, angleCab, angleAbd,
            thetaCabd, nodes, edges, interactionEdges, quadruplets, intermediateCa, intermediateDb);
    op.Process();
}
