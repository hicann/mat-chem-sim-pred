<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Model E2E Reproduction

Run `benchmark_lightgcn_movielens_e2e.py` against the official MovieLens 100K
directory and a writable checkpoint path. The script trains or loads a
64-dimensional K=2 LightGCN embedding, reports Recall@20, and benchmarks one,
two, and four disjoint graph copies with 40k/80k/160k score pairs.

The checked-in formal JSON contains checkpoint hash, component and E2E
latency, embedding/score error, binary score agreement, task metric, and
complete-model profiler evidence. Dataset and checkpoint binaries are not
stored in the source repository.
