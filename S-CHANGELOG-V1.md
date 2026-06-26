# CHANGELOG — V1

File: `dfshe_simulacrum_v1.py` (deleted)

## Parameters

| Name | Value |
|---|---|
| d_compact | 8 |
| d_common | 32 |
| n_macro | 1000 |
| k_macro | 50 |
| n_nano | 5 |
| k_nano | 2 |
| Total params | ~488,000 (estimated) |
| Params/micro | 959 |

## Architecture

- Expert storage: `nn.ModuleList` of 1000 MicroExpert modules
- Forward: Python for-loop over batch × selected experts
- Nano slot 0: NanoMLP
- Nano slot 1: NanoCNN
- Nano slot 2: NanoLinAttn
- Nano slot 3: NanoSoftTree
- Nano slot 4: NanoMLP (duplicate of slot 0)
- Routing: `gumbel_topk` function, no ST estimator

## Confirmed metrics (Colab CPU)

| Metric | Value |
|---|---|
| params/micro | 959 |
| B=64 time | 1830 ms |
| Budget violation | 1.92x over README 500 target |

## Why destroyed and forgotten?

- Budget exceeded by 1.92x (D_COMPACT=8 makes nano-zoo alone 626 params)
- CPU performance unusable (1830ms)
- Code style violations per Phase 3 feedback
