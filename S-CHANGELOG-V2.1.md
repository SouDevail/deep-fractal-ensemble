# CHANGELOG — V2.1

File: `simulacrum-V2.1.py`

## Changes from V2

| # | What | V2 | V2.1 |
|---|---|---|---|
| 1 | d_compact | 4 | 6 |
| 2 | Total params | 244,350 | 498,480 |
| 3 | Params/micro | 239 | 491 |
| 4 | Nano slot 4 | MLP (duplicate) | Elman RNN (2-step, stateless) |
| 5 | New tensors | — | rnn_ih (1000,6,6), rnn_hh (1000,6,6), rnn_b (1000,6) |
| 6 | Removed tensors | mlp1_w (1000,4,4), mlp1_b (1000,4) | — |
| 7 | gumbel_topk signature | (logits, k) | (logits, k, training=True) |
| 8 | Gumbel noise | always | only when training=True |
| 9 | Eval mode | none | deterministic topk (scores=logits) |
| 10 | cnn_proj_w shape | (1000,4,8) | (1000,6,12) |
| 11 | All (n_macro,d,d) tensors | 4×4=16 each | 6×6=36 each |
| 12 | __main__ test | params, shape, sparsity, timing | + eval_deterministic check |
| 13 | micro_w inline | m_w = self.micro_w[idx] (allocated) | inlined into einsum |
| 14 | Comment line | #nanoexperts | #nanoexperts: mlp/conv/lin-attn/soft-tree/elman-rnn |

## Confirmed metrics (Colab CPU, simv2.1-output.py)

| Metric | Value |
|---|---|
| params | 498,480 |
| per_micro | 491 |
| out shape | (4, 10) |
| mask shape | (4, 1000) |
| sparsity | 2.00% |
| eval_deterministic | True |
| b=1 time | 1.3 ms |
| b=64 time | 12.8 ms |

b=64 faster than V2 (12.8ms vs 21.6ms) despite 2x more params.

## Nano slot layout

| Slot | Archetype |
|---|---|
| 0 | MLP (relu) |
| 1 | CNN (2 filters, kernel 3, proj) |
| 2 | LinAttn |
| 3 | SoftTree (2-leaf) |
| 4 | Elman RNN (tanh, 2-step, h0=0) |
