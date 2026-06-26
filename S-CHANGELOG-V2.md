# CHANGELOG — V2

File: `simulacrum-V2.py`

## Changes from V1

| What | V1 | V2 |
|---|---|---|
| d_compact | 8 | 4 |
| d_common | 32 | 8 |
| Expert storage | nn.ModuleList | stacked nn.Parameter |
| Forward loop | Python for-loop | einsum + index_select |
| params/micro | 959 | 239 |
| B=64 time | 1830 ms | 8.9 ms |

## Parameters

| Name | Value |
|---|---|
| d_in | 64 |
| d_compact | 4 |
| d_common | 8 |
| d_out | 10 |
| n_macro | 1000 |
| k_macro | 50 |
| n_nano | 5 |
| k_nano | 2 |
| tau | 1.0 |
| Total params | 244,350 |
| Params/micro | 239 |

## Nano slot layout

| Slot | Archetype |
|---|---|
| 0 | MLP (relu, d×d) |
| 1 | CNN (2 filters, kernel 3, proj back to d) |
| 2 | LinAttn (q·(k·v) / d, no softmax) |
| 3 | SoftTree (2-leaf, softmax gate) |
| 4 | MLP (duplicate of slot 0) |

## Key parameter shapes

| Tensor | Shape | Params |
|---|---|---|
| compress | Linear(64,4) | 260 |
| macro_router | Linear(4,1000) | 5,000 |
| micro_w | (1000,5,4) | 20,000 |
| cnn_proj_w | (1000,4,8) | 32,000 |
| tree_leaf_w | (1000,2,4,4) | 32,000 |
| bridge_w | (1000,8,4) | 32,000 |
| head | Linear(8,10) | 90 |

## Confirmed metrics (Colab CPU, simv2-output.py)

| Metric | Value |
|---|---|
| params | 244,350 |
| per_micro | 239 |
| out shape | (4, 10) |
| mask shape | (4, 1000) |
| sparsity | 2.00% |
| b=1 time | 1.6 ms |
| b=64 time | 21.6 ms |

## Known gaps at V2

- No eval mode (Gumbel noise always active)
- No ST estimator (backward broken)
- No load-balance loss
- Nano slot 4 is duplicate (heterogeneity gap)
- params/micro = 239, headroom to 500 unused
