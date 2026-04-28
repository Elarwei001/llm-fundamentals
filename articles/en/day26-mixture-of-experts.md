# Day 26: Mixture of Experts (MoE) — Sparse Activation, More Model with Less Compute

> **Core Question**: How can a model with 671 billion parameters use only 37 billion during inference — and why does that work?

---

## Opening

Imagine a hospital. When a patient walks in, they don't see every doctor simultaneously — a triage nurse routes them to the right specialist. A broken arm goes to orthopedics, chest pain to cardiology, blurry vision to ophthalmology. Each doctor is highly trained, but on any given day, most are waiting. The hospital's total expertise is enormous, yet each patient only needs a few specialists.

This is exactly how Mixture of Experts (MoE) works in large language models. Instead of running every parameter for every token (like a clinic where every patient sees every doctor), MoE models contain many "expert" sub-networks but only activate a small subset per token. A router — the triage nurse — decides which experts handle each token.

The result? A model with the *knowledge capacity* of something enormous, but the *compute cost* of something much smaller. DeepSeek-V3 has 671 billion total parameters but activates only 37 billion per token — about 5.5% of its full weight. DeepSeek-V4 Pro pushes this further: 1.6 trillion total parameters, only 49 billion active. That's 3% utilization.

MoE isn't a fringe trick anymore. In 2025–2026, it has become *the* dominant architecture for frontier models. Llama 4 Maverick, Qwen3-235B, DeepSeek-V3/V4 — all MoE. If you want to understand modern LLMs, you need to understand MoE.

---

## 1. What Is Mixture of Experts?

### 1.1 From Dense to Sparse

In a standard (dense) transformer, every token passes through every parameter in the feed-forward network (FFN) at each layer. A 70B parameter model uses all 70B parameters for every single token. This is simple but wasteful — not every piece of text needs the same kind of processing.

#### Intuition: The Restaurant Kitchen

Think of a dense model as a kitchen where every chef works on every dish. MoE is a kitchen with specialized stations — the pastry chef handles desserts, the grill master handles steaks, the saucier makes sauces. Each dish only goes to the relevant stations. The kitchen has more total talent, but each order only uses a fraction of the staff.

MoE replaces the dense FFN in each transformer layer with a set of parallel expert networks plus a gating network (router) that decides which experts to use:

![MoE Architecture Overview](../zh/images/day26/moe-architecture-overview.png)
*Figure 1: MoE layer architecture. A router selects a subset of experts for each input token, while unselected experts remain inactive.*

### 1.2 Key Terminology

| Term | Definition |
|------|-----------|
| **Expert** | A feed-forward sub-network. Typically a standard 2-layer MLP. |
| **Router / Gate** | A small neural network that outputs a probability distribution over experts. |
| **Top-K routing** | Select the K experts with highest router scores (usually K=1, 2, or 8). |
| **Total parameters** | All parameters across all experts (what you need to store in memory). |
| **Active parameters** | Parameters actually used per token (what determines compute cost). |
| **Sparsity ratio** | Total / Active — how much of the model is "sleeping" per token. |

### 1.3 The Core Mechanism

For each token with hidden state $h$, the router computes:

$$
\begin{aligned}
G(h) &= \text{softmax}(W_g \cdot h) \\
\text{TopK}(G(h), K) &= \text{select K experts with highest scores}
\end{aligned}
$$

The output of the MoE layer is a weighted combination of the selected experts' outputs:

$$
y = \sum_{i \in \text{TopK}} G(h)_i \cdot E_i(h)
$$

Here, the $E$ stands for **Expert**. So $E_i(h)$ means: **the output produced by the $i$-th expert network for input $h$**.

You can read the formula piece by piece:
- $G(h)_i$ = the weight assigned by the router to the $i$-th expert
- $E_i(h)$ = the actual output computed by the $i$-th expert

So the whole equation means:

> the final output $y$ is the sum of each selected expert's output, weighted by its gate score.

In other words, the router decides *who gets activated and how much they matter*, while the experts actually do the work.

Only the selected experts perform computation. The rest sit idle.

![Expert Routing Mechanism](../zh/images/day26/expert-routing-mechanism.png)
*Figure 2: How routing works — the gate network scores each expert, selects top-K, and combines their weighted outputs.*

---

## 2. Dense vs MoE: The Fundamental Trade-off

### 2.1 Compute vs Memory

MoE creates an unusual trade-off profile:

| Aspect | Dense Model | MoE Model |
|--------|------------|-----------|
| Compute per token | Proportional to total params | Proportional to active params |
| Memory (VRAM) | Proportional to total params | Proportional to total params |
| Model quality (per FLOP) | Lower | Higher |
| Model quality (per byte) | Higher | Lower |
| Communication (multi-GPU) | Lower | Higher |

#### Intuition: The Library vs The Bookshelf

A dense model is like having a small bookshelf — every book is right there, cheap to access, but you're limited to what fits. An MoE model is like having access to a vast library — enormous knowledge, but you need to walk to the right section (routing) and the library needs floor space (memory) for all those books.

The three most confusing phrases here are usually: **"higher model quality per FLOP,"** **"lower model quality per byte,"** and **"less active compute but more communication."** Here is the intuition:

- **Model quality per FLOP** means: if you spend the same amount of computation, which model gives you better performance? MoE often wins here because each token activates only a small subset of experts, yet benefits from a much larger total parameter pool.
- **Model quality per byte** means: if you spend the same amount of memory, which model gives you more capability? Dense models often win here because almost every stored parameter gets used every time, while MoE must store many experts that are only occasionally activated.
- **Why communication is higher**: lower active compute does not mean lower data movement. The router must decide where each token goes, and if the selected experts live on different GPUs, tokens must be shuffled across devices and then gathered back afterward. That all-to-all movement can become a major systems cost.

So the bottleneck often shifts: dense models are more dominated by **raw compute**, while MoE systems are more likely to be dominated by **routing, scheduling, and cross-GPU communication**.

The key insight: **if compute is your bottleneck (which it usually is during training), MoE gives you more quality per training dollar.** If memory is your bottleneck (which it often is during deployment), MoE can be harder to serve.

![Dense vs MoE Comparison](../zh/images/day26/moe-vs-dense-comparison.png)
*Figure 3: Dense models activate all parameters per token; MoE models activate only a fraction, offering similar quality at much lower compute cost.*

### 2.2 Why MoE Works: Conditional Computation

The deep reason MoE works is *conditional computation* — different inputs get different processing paths. This is more parameter-efficient because:

1. **Specialization**: Each expert can focus on specific patterns (syntax, reasoning, factual recall, code, etc.)
2. **Capacity**: The total parameter count provides vast knowledge storage, even if each token only accesses a slice
3. **Efficiency**: Active parameters stay low, so training and inference are cheaper per token

Research from DeepSeek-V3 showed that experts don't always cleanly specialize by topic. Instead, they often specialize by *token frequency and position* — some experts handle common tokens, others handle rare ones. The model discovers useful divisions on its own.

#### Recent research: what do experts actually specialize in?

This question has become an active research topic, and a useful answer is: **experts are often real, but their specialization is usually messier and more statistical than the human-friendly story of "math expert" or "history expert."**

A few representative findings:

| Paper / project | Institution(s) | Year | What it found |
|---|---|---|---|
| **OpenMoE** | Colossal-AI / National University of Singapore collaborators | 2024 | Reported **context-independent specialization**, early routing learning, and a late-layer drop in routing diversity, suggesting experts often lock onto stable token-level patterns early in training. |
| **DeepSeekMoE** | DeepSeek-AI | 2024 | Argued for finer-grained expert specialization and introduced **shared experts**, partly because some knowledge appears too general to be cleanly partitioned into routed experts. |
| **OLMoE** | Allen Institute for AI and collaborators | 2024/2025 | Measured routing properties directly and found high specialization, low co-activation, and signs of **domain and vocabulary specialization** rather than neat textbook-style subject partitions. |
| **Soft MoE** | Google DeepMind | 2024 | Showed that softer assignment can reduce some hard-routing pathologies such as token dropping and unstable balancing, reframing specialization as a continuum rather than a strict winner-take-all assignment. |
| **Expert Choice Routing** | Google Research | 2022 | Showed that routing design itself changes which experts get used and how balanced they are, meaning observed "specialization" is partly a property of the routing rule, not just the data. |

Researchers have also started doing **expert ablation / lesion studies**: disable one expert, a group of experts, or reroute certain token families, then measure what breaks. The emerging picture is nuanced:

- Some experts really are more important than others.
- Some experts look tied to vocabulary frequency, formatting, or positional patterns.
- Some deeper experts appear more correlated with semantic or task-level behavior.
- But many functions are still distributed, so "remove one expert = remove one concept" is usually too simple.

So the current frontier view is not that MoE contains a tidy panel of human-readable specialists. It is closer to this:

> MoE learns a sparse division of labor, but that division is shaped by optimization pressure, routing design, load balancing, and data statistics, not just by human-interpretable topics.

If you want the practical takeaway, it is this: modern MoE research is shifting from merely asking **"Does sparse routing work?"** to asking **"What kind of specialization emerges, how stable is it, and can we control or interpret it?"**

> #### Research note: is MoE an engineering approximation to manifold-like structure?
>
> One useful high-level intuition is that MoE does not explicitly learn a continuous manifold, but it does seem to approximate something manifold-like in an engineering, discrete way: **different inputs occupy different local regions of representation space, so they may benefit from different local computation paths.**
>
> That resonates with the core intuition of manifold learning, because both perspectives emphasize **locality**. If the input distribution really does have local structure, then expert specialization can be viewed as a form of **piecewise, conditional function approximation** over those regions. That said, it would be too strong to call MoE a true manifold model. Its routing is discrete, system-constrained, and optimization-driven. But thinking of it as a mechanism for assigning different computational experts to different local regions is often a very productive way to understand why sparse routing works.

---

## 3. The Routing Problem, and MoE's Other Practical Challenges

### 3.1 Load Balancing

The biggest challenge in MoE is routing collapse — the tendency for the router to send all tokens to the same few experts, leaving others idle. This wastes the whole point of having multiple experts.

![Load Balancing Problem](../zh/images/day26/load-balancing-problem.png)
*Figure 4: Collapsed routing wastes expert capacity; balanced routing distributes work evenly. The auxiliary loss penalizes imbalance.*

#### Intuition: The Popular Restaurant

Imagine a food court where everyone crowds into one restaurant while others are empty. The popular kitchen overloads and slows down; the empty ones waste rent. MoE needs a system that spreads customers around.

### 3.2 Auxiliary Loss

The standard solution, introduced in Switch Transformers (Fedus et al., 2022), adds an auxiliary load-balancing loss to the training objective:

$$
L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

Where:
- $f_i$ = fraction of tokens routed to expert $i$
- $P_i$ = mean router probability for expert $i$
- $N$ = number of experts
- $\alpha$ = small coefficient (typically 0.01)

### 3.3 MoE has more challenges than load balancing

Load balancing is the most famous problem, but it is not the only one. In practice, MoE systems face a whole cluster of challenges:

| Challenge family | What can go wrong |
|---|---|
| **Routing dynamics** | Routing collapse, expert under-utilization, unstable early training, token dropping |
| **Systems cost** | Expensive all-to-all communication, cross-GPU scheduling complexity, variable latency |
| **Optimization** | Sparse routing is harder to train, auxiliary losses are sensitive, some experts learn too slowly |
| **Specialization** | Experts may specialize in shallow statistical patterns rather than robust task-level skills |
| **Serving cost** | Total parameters must still live in memory, so deployment can remain expensive |
| **Safety / control** | Different experts may not align equally well, and behavior can shift when routing changes |
| **Evaluation** | It is harder to tell whether gains come from true capability, routing quirks, or benchmark-specific effects |

#### Intuition: MoE saves compute, not complexity

A good mental model is this:
- Dense models are often bottlenecked by **raw computation**.
- MoE models reduce part of that compute.
- But in exchange, they introduce a bigger **coordination problem**.

So MoE is not just "a cheaper dense model." It is more like a distributed organization that needs routing, balancing, synchronization, and monitoring to function well.

That is why modern MoE research is split across three fronts at once:
1. **algorithmic**: better routing and better expert specialization,
2. **systems**: lower communication and smoother deployment,
3. **scientific**: understanding what experts actually learn.

This penalizes imbalance — when some experts get too many tokens (high $f_i$) and the router strongly prefers them (high $P_i$), the loss increases.

**The problem**: A large auxiliary loss interferes with the main training objective, degrading model quality. A small one may not prevent collapse. This tension is a central design challenge.

### 3.3 Auxiliary-Loss-Free Approaches

Recent work has explored alternatives. DeepSeek-V2 introduced a bias-based approach: instead of an auxiliary loss, the router's bias terms are dynamically adjusted to maintain balance. If an expert receives too many tokens, its bias is decreased; if too few, increased.

Han et al. (2025) proposed a theoretical framework for auxiliary-loss-free load balancing, showing that bias-adjustment methods can achieve balance without interfering with gradient signals from the main loss.

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| Auxiliary loss | Penalize imbalance in training objective | Simple, well-studied | Interferes with main loss |
| Bias adjustment | Dynamically adjust router bias terms | No gradient interference | More complex implementation |
| Expert choice | Experts choose tokens (not vice versa) | Perfect balance by construction | Harder to control per-token quality |
| Capacity factor | Hard limit on tokens per expert | Prevents overload | May drop tokens |

---

## 4. MoE in Practice: Major Models

### 4.1 Evolution of MoE Models

![MoE Model Evolution](../zh/images/day26/moe-model-evolution.png)
*Figure 5: MoE model evolution showing total vs active parameters. The gap between total and active parameters has grown dramatically, with DeepSeek-V4 Pro activating only 3% of its 1.6T parameters.*

### 4.2 Key MoE Models Compared

| Model | Date | Total Params | Active Params | Experts | Top-K | Notable Feature |
|-------|------|-------------|---------------|---------|-------|----------------|
| Mixtral 8x7B | Dec 2023 | 47B | 13B | 8 | 2 | First competitive open MoE LLM |
| DeepSeek-V2 | May 2024 | 236B | 21B | 160 | 6 | Shared + routed experts |
| DeepSeek-V3 | Dec 2024 | 671B | 37B | 256 | 8 | Auxiliary-loss-free routing |
| Gemma 4 26B A4B | Apr 2025 | 26B | 4B | — | — | Small, efficient MoE from Google |
| Llama 4 Maverick | Apr 2025 | 400B | 17B | 128 | — | Multimodal MoE |
| Qwen3-235B | Apr 2025 | 235B | 22B | 128 | 8 | No shared experts |
| DeepSeek-V4 Pro | Apr 2026 | 1.6T | 49B | — | — | Hybrid attention + MoE |

### 4.3 DeepSeek's Shared + Routed Expert Design

DeepSeek-V2/V3 introduced an important architectural innovation: **shared experts**. In addition to the routed experts (selected by the router), there are a few "shared" experts that process *every* token unconditionally.

#### Intuition: General Practitioners and Specialists

In a hospital, some conditions are so common (checkups, basic tests) that every patient needs them regardless of their specialist. Shared experts are like general practitioners — they handle universal processing. Routed experts are specialists — activated only when their expertise is needed.

This design reduces the burden on routed experts. Shared experts capture common patterns, leaving routed experts free to specialize more sharply.

Qwen3 takes the opposite approach — no shared experts, all 128 experts are routed with top-8 selection. The model learns to distribute knowledge without any forced sharing. Both approaches work; the field hasn't converged on a single best design.

---

## 5. Training and Serving MoE Models

### 5.1 Training Challenges

MoE models introduce unique training difficulties:

1. **Communication overhead**: In distributed training, tokens routed to experts on different GPUs require all-to-all communication. This can dominate training time.
2. **Load imbalance**: Without careful balancing, some GPUs process far more tokens than others, causing idle time.
3. **Training instability**: Sparse gradients from MoE can make training less stable than dense models.
4. **Expert underutilization**: Some experts may not receive enough training signal to learn useful functions.

#### Why are MoE gradients sparse?

In a dense model, every token flows through the same FFN, so almost every parameter in that layer receives gradient signal on every step.

In an MoE layer, each token activates only a small number of experts, such as top-2 out of 128. That means:
- only the selected experts receive gradient from that token,
- the unselected experts receive essentially nothing from it,
- and the distribution of updates depends on a router that is itself still learning.

So the sparsity comes from **conditional activation**: most expert parameters are simply not on the computation path for a given token.

This makes optimization harder because updates become:
- **uneven**: some experts see far more data than others,
- **high-variance**: one expert may receive many tokens in one batch and very few in the next,
- **self-reinforcing**: experts selected early get more gradient, improve faster, and become even more likely to be selected again.

That is one reason MoE training can be less stable than dense training, especially early on before routing has settled.

#### How do we know whether an expert is undertrained?

This is an important practical question. In a large MoE system, some experts may see huge numbers of tokens and become highly refined, while others may see too few examples and remain undertrained.

There is no single universally accepted "expert saturation" metric, but practitioners usually monitor a bundle of proxies:

| Signal | What it tells you |
|---|---|
| **Token load / routing frequency** | Whether an expert actually sees enough training data |
| **Gradient norm / update activity** | Whether its parameters are receiving meaningful learning signal |
| **Per-expert loss or perplexity** | Whether it performs well on the tokens it is assigned |
| **Routing stability** | Whether it has developed a consistent specialization over time |
| **Ablation impact** | Whether disabling that expert causes measurable degradation |

The key subtlety is that **low usage does not automatically mean low value**. Some experts may serve rare but difficult or high-value cases. So in practice, researchers look for a combination of:
- enough exposure,
- enough parameter updates,
- stable specialization,
- and evidence that the expert matters when removed.

That is why MoE monitoring is not just about balancing load. It is also about asking whether each expert is actually learning a useful role.

### 5.2 Serving Considerations

Serving MoE models requires careful engineering:

- **Memory**: You need to hold *all* parameters in memory (or fast storage), even though only a fraction is active per token
- **Batching**: Different tokens in a batch may route to different experts, requiring dynamic batching
- **Latency**: Routing adds a small overhead per layer
- **Quantization**: MoE models can be quantized (e.g., FP8 for DeepSeek-V3), but expert routing adds complexity

#### Intuition: The Warehouse

Serving a dense model is like running a small shop — everything is on display, easy to grab. Serving MoE is like running a massive warehouse — you need the whole building (memory), but each customer only visits a few aisles (active parameters). The warehouse holds more inventory (knowledge) but costs more to maintain (memory), even if each visit is quick (low compute per token).

---

## 6. Code Example: Minimal MoE Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """A minimal Mixture of Experts layer with top-K routing."""
    
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router: maps hidden state to expert probabilities
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Experts: each is a standard 2-layer FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Flatten for routing: (batch * seq, d_model)
        x_flat = x.view(-1, d_model)
        
        # Compute router scores: (batch * seq, num_experts)
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-K experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # Renormalize selected probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute weighted sum of selected expert outputs
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            for b in range(x_flat.shape[0]):
                expert_idx = top_k_indices[b, i].item()
                expert_output = self.experts[expert_idx](x_flat[b:b+1])
                output[b:b+1] += top_k_probs[b, i] * expert_output
        
        return output.view(batch_size, seq_len, d_model)

# Example usage
moe = MoELayer(d_model=512, d_ff=2048, num_experts=8, top_k=2)
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10
y = moe(x)
print(f"Input: {x.shape} -> Output: {y.shape}")
# Only 2 out of 8 experts active per token!
```

---

## 7. Common Misconceptions

### ❌ "MoE experts specialize by topic (one for code, one for math, etc.)"

Reality is more nuanced. Research shows experts often specialize by token frequency and positional patterns rather than clean semantic categories. The model discovers its own division of labor during training — and it's not always interpretable.

### ❌ "MoE is always more efficient than dense"

MoE is more compute-efficient (more quality per FLOP) but less memory-efficient. If VRAM is your bottleneck, a dense model may be more practical. MoE shines when you have enough memory but want to maximize training compute or inference throughput.

### ❌ "More experts = better"

Adding experts increases capacity but also increases routing difficulty, communication costs, and the risk of undertrained experts. Diminishing returns set in. The optimal number of experts depends on model size, data volume, and hardware.

---

## 8. Frontier: What's New in 2025–2026

### DeepSeek-V4 (April 2026)
DeepSeek-V4 Pro uses 1.6 trillion total parameters with only 49 billion active — a 3% activation ratio. Its key innovation is hybrid attention that combines compressed sparse attention (CSA) with heavily compressed attention (HCA), making million-token context practical. The MoE architecture continues DeepSeek's tradition of auxiliary-loss-free routing with bias adjustment. ([Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf))

### DeepSeek-V4-Flash (April 2026)
A companion model with 284B total / 13B active parameters, optimized for speed. Demonstrates the "intelligence density" trade-off in MoE design — how much quality per active parameter. ([Forbes Analysis](https://www.forbes.com/sites/geruiwang/2026/04/26/deepseek-v4-shows-that-the-next-ai-race-is-about-efficiency/))

### Auxiliary-Loss-Free Load Balancing (December 2025)
Han et al. formalized the theoretical framework behind bias-based load balancing, proving convergence guarantees. This validates DeepSeek's practical approach and gives the community a principled alternative to auxiliary losses. ([Paper on arXiv](https://arxiv.org/abs/2512.03915))

### MoE Survey (July 2025)
A comprehensive survey by Zhang et al. cataloged design choices across gating strategies, sparse and hierarchical variants, multimodal extensions, and deployment considerations — showing that evaluation increasingly emphasizes expert diversity and calibration. ([Survey on arXiv](https://arxiv.org/abs/2507.11181))

---

## 9. Further Reading

### Foundational Papers
1. ["Switch Transformers: Scaling to Trillion Parameter Models"](https://arxiv.org/abs/2101.03961) — Fedus et al., 2022. Introduced the simplified top-1 routing and auxiliary loss.
2. ["Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017. The original MoE paper for deep learning.
3. ["Mixture-of-Experts with Expert Choice Routing"](https://arxiv.org/abs/2202.09368) — Zhou et al., 2022. Experts choose tokens instead of tokens choosing experts.

### Model Technical Reports
4. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — Auxiliary-loss-free MoE architecture details.
5. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — 128-expert, top-8 routing design.
6. [Llama 4 Model Card](https://www.llama.com/docs/model-cards-and-prompts/llama4/) — Meta's MoE approach with 128 experts.

### Advanced Reading
7. ["A Theoretical Framework for Auxiliary-Loss-Free Load Balancing"](https://arxiv.org/abs/2512.03915) — Han et al., 2025.
8. ["Mixture of Experts in Large Language Models"](https://arxiv.org/abs/2507.11181) — Comprehensive 2025 survey.

---

## Reflection Questions

1. Why does conditional computation (routing) help more in large models than small ones? What does this tell us about how scaling interacts with architecture?
2. If MoE experts don't cleanly specialize by topic, what *does* determine their specialization? How would you design an experiment to find out?
3. MoE trades memory for compute. As hardware evolves (more memory, faster interconnects), how might this trade-off shift? Would MoE become more or less dominant?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| Mixture of Experts | Replace dense FFN with multiple sub-networks, only activate a few per token |
| Router / Gate | Small network that decides which experts to use for each token |
| Top-K routing | Select the K experts with highest gate scores (typically K=2 or 8) |
| Load balancing | Ensuring tokens distribute across experts evenly to avoid collapse |
| Auxiliary loss | Training penalty for imbalanced expert utilization |
| Shared experts | Experts that process every token (generalists alongside specialists) |
| Sparsity ratio | Total params / Active params — the key efficiency metric |
| Conditional computation | Different inputs use different computational paths |

**Key Takeaway**: MoE has become the dominant architecture for frontier LLMs because it decouples knowledge capacity from compute cost. By routing each token to a small subset of experts, models like DeepSeek-V3 (671B total, 37B active) and DeepSeek-V4 Pro (1.6T total, 49B active) achieve frontier quality at a fraction of the compute a dense model would require. The central challenges — routing, load balancing, and serving efficiency — remain active research areas, with auxiliary-loss-free methods and hybrid attention designs pushing the frontier forward.

---

*Day 26 of 60 | LLM Fundamentals*
*Word count: ~2600 | Reading time: ~13 minutes*
