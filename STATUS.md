# STATUS — DF-SHE Project

Last updated: 2026 (Sim/Release)

## Current-Simulacrum — Devailoped. MVP — In progressssss.

## Files

| File | Status |
|---|---|
| simulacrum-V2.py | Archive. Forward only, 244k params, confirmed. |
| simulacrum-V2.1.py | Archive. Forward only, 498k params, eval mode, confirmed. |
| simulacrum-V2.2.py | **DEVAILS. Clean arch, split RNN bias, 504k params. GitHub target.** |
| simulacrum-VVV.py | Archive. Mixed arch+exp. Superseded by V2.2 + exp-3. |
| simv3exp-1.py | Archive. 3 bugs found. Historical value only. |
| simv3exp-2.py | Done. TARGET_REACHED epoch 17. Confirmed. |
| simv3exp-3.py | **FINAL EXP. Hybrid Adam Self-Healing. Deep refactor. GitHub target.** |

## Architecture state

| Component | Status |
|---|---|
| gumbel_topk (forward) | Working |
| gumbel_topk (backward / ST estimator) | NOT IMPLEMENTED |
| Macro routing | Working |
| Micro routing | Working |
| Nano: MLP, CNN, LinAttn, SoftTree | Working |
| Nano: Elman RNN (split bias) | Working (VVV+) — rnn_b_ih + rnn_b_hh |
| Nano: LSTM | Not implemented |
| Eval mode | Working (V2.1+) |
| Load-balance loss | Not implemented |
| Self-Healing Hybrid Adam | Implemented (exp-3), untested — needs trigger condition |
| Training loop | Working (exp-2, exp-3) |
| Backward pass | Not smoke-tested |

## Blockers for MVP

| # | Blocker | Blocks |
|---|---|---|
| BL-1 | No ST estimator in gumbel_topk | Any real training |
| BL-2 | No load-balance loss (macro + micro) | Expert collapse prevention |
| BL-3 | Backward pass not smoke-tested | Confirming gradient flow |

## Open gaps (unresolved)

| Gap | Description |
|---|---|
| G-1 | RNN vs MLP1 accuracy NOT compared |
| G-2 | Hybrid Adam Self-Healing never triggered — effectiveness unknwn |
| G-3 | Training tsk for MVP not chosen |
| G-4 | Tau annealing schedule not chosen |
| G-5 | n_macro for MVP: stay at 1000 or scale to 20 per README? |
| G-6 | LinAttn is q*(k·v) scalar — not true attention (audit V2-P3) |
| G-7 | MOMENTUM_BOOST removed; LR_DAMPENING kept but also never triggered |

## Confirmed metrics (latest)

| Version | Params | Params/micro | Sparsity | b=1 | b=64 |
|---|---|---|---|---|---|
| V2 | 244,350 | 239 | 2.00% | 1.5ms | 8.9ms |
| V2.1 | 498,480 | 491 | 2.00% | 1.6ms | 12.0ms |
| V2.2 | 504,480 | 497 | 2.00% | 1.6ms | 12.4ms |
| exp-3 (train) | 504,480 | 497 | 2.00% | — | — | TARGET ep.15, val=0.427, 20.4s |
