# CHANGELOG — V2.2

File: `simulacrum-V2.2.py`

## Purpose

Final canonical artifact of the Simulacrum stage.
Architecture only — no training loop, no controller, no Config class.
Base for MVP stage.

## Changes from V2.1

| # | What | V2.1 | V2.2 |
|---|---|---|---|
| 1 | rnn_b | shared (1000,6) — used twice | split: rnn_b_ih + rnn_b_hh (1000,6) each |
| 2 | Total params | 498,480 | 504,480 |
| 3 | Params/micro | 491 | 497 |
| 4 | Inherited from VVV | — | F-1 RNN split bias |

## What staying still from V2.1

- All other nano archetypes: mlp0, cnn, lin-attn, soft-tree — identical
- gumbel_topk signature: (logits, k, training=True) — identical
- eval_deterministic behavior — confirmed
- Module-level lowercase constants — identical style
- __main__ block: params, shape, sparsity, eval, timing — identical structure

## Nano slot layout

| Slot | Archetype |
|---|---|
| 0 | MLP (relu) |
| 1 | CNN (2 filters, kernel 3, proj) |
| 2 | LinAttn |
| 3 | SoftTree (2-leaf) |
| 4 | Elman RNN (tanh, 2-step, split bias) |

## Confirmed metrics (Colab CPU)

| Metric | Value |
|---|---|
| params | 504,480 |
| per_micro | 497 |
| sparsity | 2.00% |
| eval_deterministic | True |
| b=1 time | 1.6ms |
| b=64 time | 12.4ms |
