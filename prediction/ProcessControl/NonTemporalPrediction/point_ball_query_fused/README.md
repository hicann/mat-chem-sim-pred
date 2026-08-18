<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# PointBallQueryFused

`PointBallQueryFused` finds up to `sample_count` source points within a radius
for every query point. It fuses squared-distance evaluation, radius filtering,
bounded selection, and valid-count output for PointNet++ neighborhood grouping.

The production evidence uses a pretrained PointNet++ SSG classifier and real
ModelNet40 point clouds. On Ascend 910B3, replacing both ball-query calls on top
of the optimized FPS path reduced model latency by 74.62%, 80.48%, and 82.90%
for batches 1, 4, and 8, with exact neighbor indices and model logits.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j
python -m pytest tests/test_point_ball_query_fused.py -q
python tests/smoke_point_ball_query_fused_npu.py --build-dir build
```

See `docs/benchmark.md` for the strong-baseline definition and complete data.
