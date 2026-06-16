# Deep Fractal Sparse Heterogeneous Ensemble  
**Neural Network System Architecture**  
*Open source project. Active edition*

---


## ⛾ 1. Fundamental Basis

Instead of a single monolithic network where parameters are mixed into huge matrices and each one contributes a little to everything, the system is built as a distributed modular structure. 

It consists of thousands (1000+) of independent, very small expert networks. 

↪ Each expert averages **500 parameters** — roughly 1–2 kilobytes in quantized or half-precision format.

Individually, no module has any power. The entire strength lies in the simultaneous parallel operation of hundreds of tiny specialists. 

Analogy — the cerebral cortex: no single super‑processor, but many highly specialized micro‑zones that activate as needed. 
Extreme modularity, thousands of micro‑experts, each knowing one thing — but knowing it precisely.


---


## ⸙ 2. Heterogeneity Principle

In standard Mixture‑of‑Experts, all experts share the same architecture and differ only in weights. 
Here a different approach is used. 

If a module is to be a true specialist for a specific task, both its internals and its form must match.

***Distribution:***
- Expert working with text sequences or code syntax — **micro‑Transformer**, **RNN**, or **LSTM**;
- Expert analyzing spatial patterns, visual layouts, geometry — **micro‑CNN**;
- Expert handling rigid mathematical checks, compliance with standards, logic triggers — **micro‑MLP** or tree‑based structure.

Inside a single ensemble, hundreds of modules with fundamentally different mathematical natures operate simultaneously. 

⚈ One parses IPv4 packets,  ⚈ another Python syntax, ⚈ a third a specific physical formula. Each performs its atomic task.


---


## ⊹ 3. Fractal Structure: 《 MoE within MoE 》

A 500‑parameter expert is not a flat single‑layer block. It is structured as a miniature Mixture‑of‑Experts. Inside the parameter budget reside:

➤ its own **micro‑router** (≈20 parameters);

➤ 4–5 **nano‑experts** with 80–100 parameters each.

Nano‑experts do not deal with high‑level semantics. They operate on atomic logical operations: binary triggers, mathematical signs, distinguishing IPv4 from IPv6, checking a specific clause of a specification.

A recursive hierarchy emerges: main router → group of micro‑experts → each micro‑expert, in turn, is a tiny MoE with its own router and several nano‑experts. This is a **fractal structure** — not in the sense of repeating connection topology, but as a two‑level composition of functions.


---


## ☲ 4. Two‑Level Sparse Activation

Compute savings are achieved by cascaded deactivation of everything unnecessary. Activation proceeds in two stages.

𝄖
**Level 1 – Macro‑routing**  
The main router analyzes the input query and selects from the entire thousand available modules only a relevant group — for example, 50 of them. The remaining 950 stay completely idle.

𝄖
**Level 2 – Micro‑routing**  
Inside each of these 50 activated micro‑experts, the local nano‑router fires. It does not activate all 4–5 nano‑experts, but only the 1–2 actually needed to process the current signal. The rest again sleep.
𝄖


➢ [*Result*] : 

𝄃 For processing a complex query, the system simultaneously uses only a few thousand parameters out of potentially hundreds of thousands. This enables inference and even fine‑tuning on ordinary consumer hardware — no GPU, no cloud, no large budget.


---


## ♆ 5. Scientific and Engineering Challenges

**Differentiable Routing via Cascade**  
*Routers make discrete decisions:* send the signal ➥ to ➥ expert #5, not #7 ⌿ . 

Discrete choice breaks backpropagation — the gradient does not pass through a hard threshold. 
The solution is to use differentiable approximations: **Gumbel‑Softmax** or Straight‑Through Estimator. 

With two cascade levels, the gradient must pass through two such approximations sequentially, which can accumulate noise and destabilize training. This requires careful ***temperature annealing*** and, possibly, ***independent Straight‑Through Gumbel‑Softmax*** on each router followed by *gradient summation*.

**Load Balancing**  
🠚 In any MoE, routers tend to collapse onto a few “favorite” experts, leaving others idle. An auxiliary load‑balancing loss is mandatory, penalizing imbalances and forcing uniform expert utilization. In the two‑level system, this loss must be applied at both macro and micro levels.

**Embedding Bridges**  
🠚 Heterogeneous experts produce outputs of fundamentally different formats: feature maps from CNNs, sequence embeddings from Transformers, flat vectors from MLPs. 

Before aggregation, they must be brought to a common denominator. Lightweight projection layers — **adapters** (*a single linear layer or a tiny two‑layer MLP*) are used. 
Each such bridge adds only 10–20 parameters per connection and learns to align the output semantics to the common space.


---


## ❦ 6. Application Areas

The architecture scales in both directions:

- ︶ **Down** — running on microcontrollers, edge devices, IoT. Local anomaly detection, traffic inspection, offline expert systems.
---
- ﹋ **Up** — with enough experts, quality data, and engineering resources, the ensemble can theoretically reach the performance level of modern giants on specialized benchmarks, while retaining modularity and the ability to fine‑tune without full retraining.

**〔 A Step Toward Human‑Like Thinking {/} 〕**  
A modular heterogeneous architecture is one path to flexible, economical intelligence that can switch between drastically different tasks without full brain rewiring.

Human intelligence is not monolithic; it consists of many specialized mechanisms that activate cascadingly with extreme energy efficiency. Implementing this scheme yields a system organized like a living brain: not a single black box, but a coordinated  orchestra  of tiny “why‑and‑how” units, each about its own domain. Modularity, fine‑grained specialization, and sparse activation are the keys to the “humanness” of behavior that large labs are chasing.

Other niches: *cybersecurity, compliance checking, on‑device personal assistants, continuous learning systems.*


---


## 7. Implementation Roadmap

**Simulacrum**  
A script that creates a fractal graph with random weights and runs tensors through both routing levels. No training. Goal: verify dimension compatibility and graph logic.

---

**MVP**  
Scale down to 20 micro‑experts, pick a simple task (e.g., text classification). Implement Gumbel‑Softmax routing and embedding bridges. Debug end‑to‑end backpropagation.

---

**Alpha**  
Scale up to 200+ experts, target a domain such as network security. Use synthetic data based on RFCs and CVEs.

---

*We start the first item this week.*

---

## Appendix A. Architecture Diagram (ASCII)

INPUT (text, image, binary data)

|

[MAIN MACRO‑ROUTER] --- selects a group of micro‑experts

|

+-- Micro‑expert #1 (CNN, ~500 params)

| |

| [Internal micro‑router] (~20 params)

| |

| +-- Nano‑expert A (~100 params) --+

| +-- Nano‑expert B (~100 params) --|--> output #1

| |

+-- Micro‑expert #2 (Transformer) ------|

+-- Micro‑expert #3 (MLP) --------------|

... |

+-- Micro‑expert #N (LSTM) ------------|

|

[AGGREGATION LAYER] (embedding bridges + combination)

|

OUTPUT (class, answer, action)



***For clarity, 3–4 nano‑experts per micro‑module are shown. Actual count inside each is 4–5.***


---


## Appendix B. Technical Notes

**Framework:** PyTorch. Use `nn.ModuleList` and `nn.ModuleDict` to manage the dynamic set of experts.

**TinyExpert pseudocode (sketch):**
```python
class TinyExpert(nn.Module):
    def __init__(self, arch='CNN', in_dim=64, out_dim=64):
        self.router = nn.Linear(in_dim, 4)   # 4 nano‑experts
        self.nano_experts = nn.ModuleList([ NanoExpert(arch, ...) for _ in range(4) ])
    def forward(self, x):
        weights = F.gumbel_softmax(self.router(x), hard=True)
        # route to top‑2 nano‑experts, combine outputs 
        
```
----> [-1-] Conceptual skeleton. Implementation details (load balancing, annealing) require careful engineering.

--> [-2-] Embedding bridges: A single linear layer or a tiny two‑layer MLP per heterogeneous connection. Projects the expert’s output into the common aggregation space.

---

***⌗ Architecture open for implementation. README will be updated if there a big changes. Official Devailopment is here.  ⌗***
