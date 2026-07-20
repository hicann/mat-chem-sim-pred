#!/usr/bin/env python3
"""Build driver for the BatchSpdInvFp32 custom op (pure-Vector batched SPD inverse).

No Cube / Matmul API: tiling carries only {batch, m}; batches are split across
AIV cores. Run on node202 after sourcing the CANN set_env.sh.
"""
import glob
import os
import re
import shutil
import subprocess

REPO = "/home/ql2025/work/tslib_cann_ops_dev"
JSON = os.path.join(REPO, "batch_spd_inv_fp32_msopgen.json")
KERNEL_SRC = os.path.join(REPO, "batch_spd_inv_fp32_ascendc.cpp")
OUT = os.path.join(REPO, "build/msopgen_batch_spd_inv_fp32")
COMPUTE_UNIT = "ai_core-ascend910b"

TILING_HEADER = '''#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BatchSpdInvFp32TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, m);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchSpdInvFp32, BatchSpdInvFp32TilingData)
}
'''

TILING_FUNC = '''static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  BatchSpdInvFp32TilingData tiling;
  const auto* g_shape = context->GetInputShape(0);
  const auto& g = g_shape->GetStorageShape();
  size_t nd = g.GetDimNum();
  uint32_t m = static_cast<uint32_t>(g.GetDim(nd - 1));
  uint32_t batch = 1;
  for (size_t i = 0; i + 2 < nd; ++i) {
    batch *= static_cast<uint32_t>(g.GetDim(i));
  }
  tiling.set_batch(batch);
  tiling.set_m(m);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t aiv = ascendcPlatform.GetCoreNumAiv();
  uint32_t blockDim = (batch < aiv) ? batch : aiv;
  if (blockDim == 0) blockDim = 1;
  context->SetBlockDim(blockDim);

  size_t sysWs = static_cast<size_t>(16) * 1024 * 1024;
  size_t* ws = context->GetWorkspaceSizes(1);
  ws[0] = sysWs;

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}'''

INFER_SHAPE = '''static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* g_shape = context->GetInputShape(0);
    gert::Shape* gi_shape = context->GetOutputShape(0);
    *gi_shape = *g_shape;
    return GRAPH_SUCCESS;
}'''

INFER_DTYPE = '''static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
context->SetOutputDataType(0, context->GetInputDataType(0));
return ge::GRAPH_SUCCESS;
}'''


def run(cmd, cwd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    assert os.path.isfile(JSON), JSON
    assert os.path.isfile(KERNEL_SRC), KERNEL_SRC
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    run(["msopgen", "gen", "-i", JSON, "-f", "aclnn", "-c", COMPUTE_UNIT,
         "-out", OUT, "-lan", "cpp"], cwd=REPO)

    kfiles = glob.glob(os.path.join(OUT, "op_kernel", "*.cpp"))
    assert len(kfiles) == 1, kfiles
    target_kernel = kfiles[0]
    print("generated kernel:", target_kernel)
    shutil.copyfile(KERNEL_SRC, target_kernel)

    base = os.path.basename(target_kernel)
    host_cpp = os.path.join(OUT, "op_host", base)
    host_tiling = os.path.join(OUT, "op_host", base[:-4] + "_tiling.h")
    assert os.path.isfile(host_cpp), host_cpp
    assert os.path.isfile(host_tiling), host_tiling

    with open(host_tiling, "w", encoding="utf-8") as f:
        f.write(TILING_HEADER)

    host = open(host_cpp, encoding="utf-8").read()
    if "tiling/platform/platform_ascendc.h" not in host:
        host = host.replace('#include "register/op_def_registry.h"',
                            '#include "register/op_def_registry.h"\n#include "tiling/platform/platform_ascendc.h"', 1)
    host = re.sub(r"static ge::graphStatus TilingFunc\(gert::TilingContext\* context\)\n\{.*?\n\}",
                  TILING_FUNC, host, count=1, flags=re.S)
    assert TILING_FUNC in host, "TilingFunc patch failed"
    host = re.sub(r"static ge::graphStatus InferShape\(gert::InferShapeContext\* context\)\n\{.*?\n\}",
                  INFER_SHAPE, host, count=1, flags=re.S)
    assert INFER_SHAPE in host, "InferShape patch failed"
    host = re.sub(r"static ge::graphStatus InferDataType\(gert::InferDataTypeContext \*context\)\n\{.*?\n\}",
                  INFER_DTYPE, host, count=1, flags=re.S)
    assert INFER_DTYPE in host, "InferDataType patch failed"
    host = host.replace('this->AICore().AddConfig("ascend910");', 'this->AICore().AddConfig("ascend910b");')
    with open(host_cpp, "w", encoding="utf-8") as f:
        f.write(host)

    cfg = os.path.join(OUT, "cmake", "config.cmake")
    if os.path.isfile(cfg):
        txt = open(cfg, encoding="utf-8").read()
        txt = txt.replace("ascend910 CACHE STRING", "ascend910b CACHE STRING")
        open(cfg, "w", encoding="utf-8").write(txt)

    print("==> Building BatchSpdInvFp32", flush=True)
    run(["bash", "build.sh"], cwd=OUT)
    print("BUILD_OK")


if __name__ == "__main__":
    main()
