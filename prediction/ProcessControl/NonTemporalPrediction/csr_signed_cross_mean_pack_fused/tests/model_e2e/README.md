<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Model Evidence

The scripts train and benchmark maintained PyG `SignedGCN` on Bitcoin-OTC.

Source anchor:

```text
pytorch_geometric@003c3cd8a10520567ceaeda619f0315e30ec2f66
torch_geometric/nn/conv/signed_conv.py
SignedConv.forward, first_aggr=False
```

Evidence files:

- `signedgcn_bitcoin_e2e.json`: 100 alternating-order samples at three scales,
  exact before/after latency, parity, task metric, and checkpoint hash.
- `signedgcn_bitcoin_e2e_repeat2.json`: independent final-code repeat used for
  conservative qualification.
- `signedgcn_bitcoin_hotspot.json`: complete-model Level1 kernel totals and
  exact selected framework calls.

Checkpoint SHA-256:
`2eb1361ab6c69c240d84eb109011bbff908b2d40efa94ce01f0f91bfbcc85b93`.

```bash
python benchmark_signedgcn_bitcoin_e2e.py \
  --checkpoint /path/to/signedgcn_bitcoin.pt \
  --build /path/to/build \
  --output signedgcn_bitcoin_e2e.json \
  --copies 1 4 16 --warmup 15 --repeat 100
```

The dataset, checkpoint, profiler trace directories, shared libraries, and build
trees are deliberately not committed.
