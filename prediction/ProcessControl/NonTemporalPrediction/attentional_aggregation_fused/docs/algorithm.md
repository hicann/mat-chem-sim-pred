<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

## PyG Mapping

The exact maintained call is torch_geometric.nn.aggr.AttentionalAggregation.forward.
For node feature x_i and scalar gate-logit a_i produced by PyG's unchanged gate
network, the call performs attention pooling independently for each graph.

The custom operator begins after the gate network and receives the graph
offsets, feature matrix, and scalar gate logits. It does not change the
learnable gate function, graph convolution layers, classifier, parameters, or
checkpoint format.

## Mathematical Definition

For graph g containing nodes I_g, define:

~~~
m_g = max_{i in I_g}(a_i)
z_g = sum_{i in I_g} exp(a_i - m_g)
w_i = exp(a_i - m_g) / z_g
y_g[c] = sum_{i in I_g} w_i * x_i[c]
~~~

Subtracting m_g is the required stable-softmax form. It is mathematically
equivalent to PyG attention pooling for finite inputs and prevents exponential
overflow.

## Ascend C Execution

One AICore block owns one graph. The kernel makes three local passes over that
graph:

1. Find the maximum gate logit.
2. Form the stable softmax denominator.
3. Accumulate the weighted feature vector.

Feature channels are processed in a bounded local accumulator vector. This
removes materialized node weights and intermediate scatter reductions from the
pooling stage. Each block writes exactly one non-overlapping [C] output row.

## Boundary and Fallback

The custom path is selected only when:

~~~
finite FP32 features and gates
graph_ptr[0] == 0
graph_ptr[G] == N
strictly increasing graph_ptr
G >= 4
1 <= C <= 1024
~~~

The Host C ABI validates pointers and dimensions representable by its tiling
record. The integration validates logical shapes, CSR monotonicity, and finite
gate values before selecting the custom path; otherwise it calls the native
PyG implementation. This preserves PyG behavior for empty graphs, unsupported
channel counts, and non-finite input.
