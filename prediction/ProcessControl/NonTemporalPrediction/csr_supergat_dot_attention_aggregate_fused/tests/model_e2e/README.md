# Model E2E

The consolidated benchmark `../../../benchmark_new_prediction_top5_models_e2e.py`
runs this op inside a same-weight PyG `SuperGATConv(attention_type="SD")` model
on a Cora x4 disjoint batch. The 20-sample result and TorchAir probe are in
`../../../new_prediction_top5_evidence/`.
