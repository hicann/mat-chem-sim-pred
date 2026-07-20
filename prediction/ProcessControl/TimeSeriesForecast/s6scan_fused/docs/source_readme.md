# S6 Selective Scan Fused (Experimental)
## Operator
S6 Selective Scan (Mamba-style, Gu & Dao 2023)
delta = softplus(Wd*x + bd)
A_disc = exp(-delta * a_param)
B_gate = sigmoid(Wb*x + bb)
x_proj = Wx*x + bx
h = A_disc * h + B_gate * x_proj
Weight: (3*IN, H), Bias: (4*H) -- includes a_param

## Target: Ascend 910B3
### Gate 1: Probe - max_diff 2.24e-8 PASS
### Gate 2: E2E - max_diff 5.22e-8 PASS
### Gate 3: Performance - 136.56x component, 44.15x E2E PASS
