<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# PPI FiLM model E2E

`benchmark_film_ppi_e2e.py` loads the real PPI training split and the exact
four-layer topology from maintained PyG `examples/film.py`. It compares
official PyG, an exact resident implementation with cached static degree, and
the same model with all four compatible aggregation stages replaced.

The deterministic base input is the first two-graph batch. Disconnected copies
scale it to four and eight graphs without changing degree or feature
distribution. The first three operator calls enable message ReLU; the final
121-channel call disables it.

The checkpoint is external to the source tree:

```text
/home/huawei/hot_model_15_20260801/film_ppi_50e_safe_20260802.pt
SHA256 641cc810fefb4c07d2c64220f1ab795479cecab6cbe4c93e1fe133afaadc1bca
```

The JSON files contain every timing sample, task metric, numerical comparison,
and profiler kernel breakdown.
