/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../../acl_smoke_helpers.h"
#include "gemnet_quadruplet_geometry_fused_host.h"

using MolecularGeometrySmoke::Check;
using MolecularGeometrySmoke::CopyIn;

bool RunGeometryKernel(const std::vector<void*>& inputs, void* dCab, void* dAbd, void* dTheta, aclrtStream stream,
                       float& angleCab, float& angleAbd, float& theta)
{
    const uint64_t workspace = aclnnGemNetQuadrupletGeometryFusedGetWorkspaceSize(4, 4, 1, 1, 1, 1);
    const int32_t status = aclnnGemNetQuadrupletGeometryFused(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9],
        inputs[10], dCab, dAbd, dTheta, 4, 4, 1, 1, 1, 1, nullptr, workspace, stream);
    if (status != ACL_SUCCESS) return false;
    Check(aclrtSynchronizeStream(stream), "sync");
    Check(aclrtMemcpy(&angleCab, sizeof(float), dCab, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), "cab out");
    Check(aclrtMemcpy(&angleAbd, sizeof(float), dAbd, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), "abd out");
    Check(aclrtMemcpy(&theta, sizeof(float), dTheta, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), "theta out");
    return true;
}

void Cleanup(int device, aclrtStream stream, std::vector<void*>& inputs, void* dCab, void* dAbd, void* dTheta)
{
    aclrtFree(dTheta);
    aclrtFree(dAbd);
    aclrtFree(dCab);
    for (void* input : inputs) aclrtFree(input);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
}

int RunSmoke(int device)
{
    Check(aclInit(nullptr), "init");
    Check(aclrtSetDevice(device), "device");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "stream");
    const std::vector<float> position{0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.};
    const std::vector<int32_t> source{1, 2, 0, 3};
    const std::vector<int32_t> target{0, 0, 1, 1};
    const std::vector<int32_t> interactionSource{3}, interactionTarget{0};
    const std::vector<int32_t> reduceCa{0}, expandDb{3};
    const std::vector<int32_t> reduceIntermediateCa{0}, expandIntermediateDb{3};
    const std::vector<int32_t> reduceIntermediateAb{0}, expandIntermediateAb{0};
    float angleCab = -1.F, angleAbd = -1.F, theta = -1.F;
    std::vector<void*> inputs(11, nullptr);
    const std::vector<const void*> hosts{position.data(),
                                         source.data(),
                                         target.data(),
                                         interactionSource.data(),
                                         interactionTarget.data(),
                                         reduceCa.data(),
                                         expandDb.data(),
                                         reduceIntermediateCa.data(),
                                         expandIntermediateDb.data(),
                                         reduceIntermediateAb.data(),
                                         expandIntermediateAb.data()};
    const std::vector<size_t> bytes{position.size() * sizeof(float),
                                    source.size() * sizeof(int32_t),
                                    target.size() * sizeof(int32_t),
                                    interactionSource.size() * sizeof(int32_t),
                                    interactionTarget.size() * sizeof(int32_t),
                                    sizeof(int32_t),
                                    sizeof(int32_t),
                                    sizeof(int32_t),
                                    sizeof(int32_t),
                                    sizeof(int32_t),
                                    sizeof(int32_t)};
    for (size_t i = 0; i < inputs.size(); ++i) CopyIn(&inputs[i], hosts[i], bytes[i]);
    void *dCab = nullptr, *dAbd = nullptr, *dTheta = nullptr;
    Check(aclrtMalloc(&dCab, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "cab");
    Check(aclrtMalloc(&dAbd, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "abd");
    Check(aclrtMalloc(&dTheta, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), "theta");
    if (!RunGeometryKernel(inputs, dCab, dAbd, dTheta, stream, angleCab, angleAbd, theta)) return 2;
    const bool passed = std::isfinite(angleCab) && std::isfinite(angleAbd) && std::isfinite(theta) &&
                        std::fabs(angleCab - 1.5707963F) < 2.0e-4F;
    Cleanup(device, stream, inputs, dCab, dAbd, dTheta);
    return passed ? 0 : 10;
}

int main(int argc, char** argv) { return RunSmoke(argc > 1 ? std::atoi(argv[1]) : 0); }
