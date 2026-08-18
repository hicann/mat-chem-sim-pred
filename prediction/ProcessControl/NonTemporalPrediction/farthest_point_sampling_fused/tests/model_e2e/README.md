<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# PointNet++ Model E2E

`benchmark_pointnet2_fps_e2e.py` replaces both FPS calls in a complete
PointNet++ SSG classification forward. It measures the combined FPS stage,
full-model latency, in-model indices, output parity, and sample accuracy.

External reproducibility inputs:

- PointNet++ source commit: `eb64fe0b4c24055559cea26299cb485dcb43d8dd`
- Pretrained checkpoint SHA256:
  `736a7136104fee7e92de99516bfd915c8b66b1c5432955dc37aa07f5808f6635`
- `airplane_0627/coord.npy` SHA256:
  `0a29fcc1642681650a1ccac21e06da45491375e28cce8222e18a5f3de7d6e019`
- `bathtub_0107/coord.npy` SHA256:
  `1709b83384e5419f75fabc1eaebbdc6fe2f9046ef81b86c8432fb37a61a992bc`
- Checked-in result JSON SHA256:
  `2d99243c98cb0a283f53b3ce40105bb2f33ad27eab1f35108f4555c47f76c11d`

The model source and checkpoint come from
`yanx27/Pointnet_Pointnet2_pytorch`. The point clouds are individual ModelNet40
test samples from `GaussianWorld/modelnet_processed`; external model/data files
are not vendored into this operator package.

After building the operator:

```bash
python3 tests/model_e2e/benchmark_pointnet2_fps_e2e.py \
  --pointnet2-source /path/to/pointnet2/models \
  --source-commit eb64fe0b4c24055559cea26299cb485dcb43d8dd \
  --checkpoint /path/to/best_model.pth \
  --point-clouds /path/to/airplane_0627_coord.npy /path/to/bathtub_0107_coord.npy \
  --labels 0 1 \
  --operator-build-dir build \
  --output tests/model_e2e/results/pointnet2_fps_e2e.json \
  --batches 1 4 8 --warmup 3 --repeat 10 --trials 2
```

The checked-in JSON preserves both trials and every synchronized sample. It is
a two-sample parity and latency gate, not a full ModelNet40 accuracy run.
