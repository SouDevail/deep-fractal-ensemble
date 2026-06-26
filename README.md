# Deep Fractal Sparse Heterogeneous Ensemble (DF-SHE)

Devails. AI News on Intel(R) Xeon(R) CPU @ 2.20 GHz with 2 virtual cores.

---

## 1. Fundamental Basis

Instead of a single monolithic network where parameters are mixed into huge matrices and each one contributes a little to everything, the system is built as a distributed modular structure.

It consists of thousands (1000+) of independent, very small expert networks.

Each expert averages **500 parameters** — roughly 1–2 kilobytes in quantized or half-precision format.

Individually, no module has any power. The entire strength lies in the simultaneous parallel operation of hundreds of tiny specialists.

**the cerebral cortex:** no single super-processor, but many highly specialized micro-zones that activate as needed. Extreme modularity, thousands of micro-experts, each knowing one thing — but knowing it precisely.

---

## 2. Heterogeneity Principle

standard Mixture-of-Experts: *all experts share the same architecture and differ only in weights.* Here a different approach is used.

If a module is to be a true specialist for a specific task, both its internals and its form must match.

**Distribution:**
- Expert working with text sequences or code syntax — micro-Transformer, RNN, or LSTM
- Expert analyzing spatial patterns, visual layouts, geometry — micro-CNN
- Expert handling rigid mathematical checks, compliance with standards, logic triggers — micro-MLP or tree-based structure

Inside a single ensemble, hundreds of modules with fundamentally different mathematical natures operate simultaneously.

One parses IPv4 packets, another Python syntax, a third a specific physical formula. Each performs its atomic task.

---

## 3. Fractal Structure: MoE within MoE

A 500-parameter expert is not a flat single-layer block. It is structured as a miniature Mixture-of-Experts. Inside the parameter budget reside:

- Its own micro-router (~20 parameters)
- 4–5 nano-experts with 80–100 parameters each

Nano-experts do not deal with high-level semantics. They operate on atomic logical operations: binary triggers, mathematical signs, distinguishing IPv4 from IPv6, checking a specific clause of a specification.

A recursive hierarchy emerges: **main router → group of micro-experts → each micro-expert, in turn, is a tiny MoE with its own router and several nano-experts.** This is a fractal structure — not in the sense of repeating connection topology, but as a two-level composition of functions.

---

## 4. Two-Level Sparse Activation

Compute savings are achieved by cascaded deactivation of everything unnecessary.

**Level 1 – Macro-routing:** The main router analyzes the input query and selects from the entire thousand available modules only a relevant group — for example, 50 of them. The remaining 950 stay completely idle.

**Level 2 – Micro-routing:** Inside each of the 50 activated micro-experts, the local nano-router fires. It does not activate all 4–5 nano-experts, but only the 1–2 actually needed. The rest sleep.

**Result:** For processing a complex query, the system simultaneously uses only a few thousand parameters out of potentially hundreds of thousands. This enables inference and even fine-tuning on ordinary consumer hardware — no GPU, no cloud, no large budget.

---

## 5. Scientific and Engineering Challenges

**Differentiable Routing via Cascade**
Routers make discrete decisions. Discrete choice breaks backpropagation. The solution: differentiable approximations — Gumbel-Softmax or Straight-Through Estimator. With two cascade levels, the gradient must pass through two such approximations sequentially, requiring careful temperature annealing.

**Load Balancing**
In any MoE, routers tend to collapse onto a few "favorite" experts. An auxiliary load-balancing loss is mandatory, applied at both macro and micro levels.

**Embedding Bridges**
Heterogeneous experts produce outputs of fundamentally different formats. Lightweight projection layers — adapters (a single linear layer or tiny MLP) — bring outputs to a common space. Each bridge adds only 10–20 parameters per connection.

---

## 6. Application Areas

- **Down** — running on microcontrollers, edge devices, IoT. Local anomaly detection, traffic inspection, offline expert systems.
- **Up** — with enough experts, quality data, and engineering resources, the ensemble can theoretically reach the performance level of modern giants on specialized benchmarks, while retaining modularity and the ability to fine-tune without full retraining.

Other niches: cybersecurity, compliance checking, on-device personal assistants, continuous learning systems.

---

## 7. Implementation Roadmap

| Stage | Status | Description |
|---|---|---|
| **Simulacrum** | Devail-COMPLETE | Forward pass, dimension verification, fractal graph logic, routing at both levels |
| **MVP** | [] Next | Scale to 20 experts, text classification, Gumbel-Softmax routing, end-to-end backprop |
| **Alpha** | [] Planned | 200+ experts, domain targeting, synthetic RFC/CVE data |

---

## 8. Architecture Diagram

```
INPUT (text, image, binary data)
         |
  [MACRO-ROUTER]  —— selects k=50 of 1000 micro-experts
         |
  +-- Micro-expert #1 (CNN)
  |        |
  |   [micro-router]
  |        |
  |   +-- Nano A (MLP)    --|
  |   +-- Nano B (RNN)    --+--> output #1
  |
  +-- Micro-expert #2 (Transformer) --> output #2
  +-- Micro-expert #3 (SoftTree)   --> output #3
  ...
         |
  [AGGREGATION] (embedding bridges + weighted sum)
         |
       OUTPUT
```

---

## Current Simulacrum Stage:

### Architecture (final V2.2.2026)

| Parameter | Value |
|---|---|
| d_in | 64 |
| d_compact | 6 |
| d_common | 8 |
| d_out | 10 |
| n_macro | 1000 |
| k_macro | 50 (active per forward) |
| n_nano | 5 |
| k_nano | 2 (active per micro-expert) |
| Total params | 504,480 |
| Params/micro | 497 |
| Sparsity | 2.00% |

### Nano-expert zoo (per micro-expert)

| Slot | Type |
|---|---|
| 0 | MLP (ReLU) |
| 1 | CNN (2 filters, kernel 3) |
| 2 | Linear Attention |
| 3 | Soft Decision Tree (2-leaf) |
| 4 | Elman RNN (2-step, split bias) |

### Confirmed metrics

| Version | Params | Params/micro | b=1 | b=64 |
|---|---|---|---|---|
| V2 | 244,350 | 239 | 1.5ms | 8.9ms |
| V2.1 | 498,480 | 491 | 1.6ms | 12.0ms |
| V2.2 | 504,480 | 497 | 1.6ms | 12.4ms |

### Training results (Simulacrum experiments)

| Experiment | val_loss | TARGET | Time |
|---|---|---|---|
| exp-1 | 13.285 | ⊗ | 60.9s |
| exp-2 | 0.448 | 🅥 epoch 17 | 23.8s |
| exp-3 (Hybrid Adam) | 0.427 | 🅥 epoch 15 | 20.4s |

---

## Repository Structure

```
/
├── README.md
├── STATUS.md          — current project state
├── HISTORY.md         — DEVAILopment phase history
└── simulacrum/
    ├── simulacrum-V2.py       — archive
    ├── simulacrum-V2.1.py     — archive
    ├── simulacrum-V2.2.py     — FINAL sim. (1)
    ├── changelogs/
    └── experiments/
        ├── simv3exp-1.py      — Self-Healing v1 (bugs found)
        ├── simv3exp-2.py      — Self-Healing v2 (confirmed)
        ├── simv3exp-3.py      — Hybrid Adam Self-Healing (final)
        └── changelogs/
```

---

## Self-Healing Mechanism (custom)

`simv3exp-3.py` implements a non-standard training controller: when loss rises for 3 consecutive epochs, it triggers a **hybrid Adam state surgery**:

1. `betas[0] *= 0.85` — dampens future gradient momentum
2. `exp_avg *= 0.3` for `compress` and `macro_router` parameters — partially resets accumulated gradient history on routing layers

This forces the router to recalibrate without full optimizer reset. No analogues in standard libraries.

---

## Dependencies

```
torch
```

Apache 2.0 compatible.

---

## Status

See [STATUS.md](STATUS.md) for full architecture state, blockers for MVP, and open gaps.
