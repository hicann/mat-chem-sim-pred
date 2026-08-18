<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# PointNet++ Ball-Query Reproduction

Build this operator and sibling `farthest_point_sampling_fused`, then run on an
NPU host with the PointNet++ source, pretrained checkpoint, and real point-cloud
files:

```bash
python benchmark_pointnet2_ball_query_e2e.py \
  --fps-benchmark ../../../farthest_point_sampling_fused/tests/model_e2e/benchmark_pointnet2_fps_e2e.py \
  --pointnet2-source /path/to/pointnet2 --checkpoint /path/to/best_model.pth \
  --point-clouds /path/to/sample1.npy /path/to/sample2.npy \
  --fps-build-dir ../../../farthest_point_sampling_fused/build \
  --operator-build-dir ../../build --output results/pointnet2_ball_query.json
```

The checked-in formal result is `results/pointnet2_ball_query_formal.json`.
It uses two trials, 3 warmups, and 10 synchronized samples per scale.
