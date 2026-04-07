# Day 11: Pre-training — How LLMs Learn to Model the World

> **Core Question**: How does a model go from random noise to understanding language, just by reading the internet?

---

## Opening

Imagine you're dropped into a massive library with billions of books, and your only task is: *predict the next word on every page*. No teacher, no labels, no instructions — just predict, predict, predict. After reading enough pages, something remarkable happens: you start to understand grammar, facts, reasoning patterns, even humor.

This is essentially what pre-training does. It's the single most expensive, most important phase of building an LLM — the process that turns random parameter initialization into something that genuinely understands language. GPT-4, Llama, Gemini — every major LLM went through this phase, consuming trillions of tokens on thousands of GPUs for months.

Think of pre-training like raising a child through sheer exposure to books. You don't sit down and explicitly teach grammar rules or geography facts. Instead, the child reads millions of pages and *absorbs* the patterns. Pre-training is the "reading millions of pages" part. The alignment phase (which we'll cover in Day 13–15) is the "teaching manners" part.

---

## 1. What Is Pre-training?

**Pre-training** is the first major training phase where a language model learns from massive amounts of **unlabeled text** using a **self-supervised** objective. The model teaches itself by solving prediction tasks that require no human annotation.

### 1.1 Self-Supervised Learning — The Key Insight

Traditional machine learning needs labeled data: images tagged "cat" or "dog," emails marked "spam" or "ham." This is expensive and doesn't scale. Self-supervised learning is a clever trick: **the labels are already inside the data itself**.

For a sentence like "The cat sat on the ___," the next word "mat" serves as its own label. No human annotator needed. Every piece of text on the internet contains millions of such implicit training examples.

This is why pre-training can use the entire internet — because we don't need anyone to label it. The text *is* the label.

### 1.2 The Three Pre-training Objectives

![Figure 1: Three pre-training objectives — Causal LM, Masked LM, and Prefix LM](../zh/images/day11/pretraining-objectives.png)
*Figure 1: The three main pre-training objectives. Orange tokens are prediction targets; green/blue tokens are visible context.*

**Causal Language Modeling (CLM)** — used by GPT family:
- Predict the next token given all previous tokens
- $P(x_t \mid x_1, x_2, \ldots, x_{t-1})$
- Autoregressive: can only see the past, not the future
- Optimized for text *generation*

**Masked Language Modeling (MLM)** — used by BERT:
- Randomly mask ~15% of tokens, predict them from context (both directions)
- $P(x_{\text{masked}} \mid x_{\text{surrounding}})$
- Bidirectional: sees both left and right context
- Optimized for text *understanding*

**Prefix Language Modeling (Prefix LM)** — used by T5:
- Split input into prefix (visible) and target (predict)
- Like CLM but with a bidirectional prefix section
- Unifies understanding and generation

Modern LLMs (GPT-4, Llama, etc.) almost exclusively use **Causal LM** because it naturally produces generation-capable models.

---

## 2. The Training Objective in Detail

### 2.1 The Loss Function

The core idea is remarkably simple: **predict the next word**. Given a sequence of tokens $x_1, x_2, \ldots, x_{t-1}$, the model learns to predict $x_t$. The pre-training objective is **cross-entropy loss** (equivalent to negative log-likelihood):

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1};\, \theta)
$$

$$
= -\frac{1}{T}\sum_{t=1}^{T} \log \,\text{softmax}(z_t)[x_t]
$$

Where $\theta$ are the model parameters, $z_t$ is the logit vector at position $t$, and $x_t$ is the actual next token. Lower loss means better prediction — the model is more "surprised" less often.

### 2.2 Why This Works So Well

The next-token prediction objective implicitly teaches the model:

| What it learns | How it's learned |
|----------------|------------------|
| **Grammar** | Can't predict "She go to store" correctly |
| **Facts** | "The capital of France is ___" requires knowing Paris |
| **Reasoning** | "2 + 2 = ___" requires arithmetic |
| **Common sense** | "The ice cream melted because it was ___" requires physics |
| **Style** | Different contexts need different completions |

The beauty is that we don't need labeled data — the text itself provides the supervision.

### 2.3 The Training Loop

At its core, pre-training is surprisingly simple:

```python
# Simplified pre-training loop (Causal LM)
model = TransformerLM(vocab_size=128000, dim=4096, n_layers=32)

for batch in dataloader:  # Each batch: sequences of token IDs
    inputs = batch[:, :-1]   # All tokens except last
    targets = batch[:, 1:]   # All tokens shifted by 1 (next-token targets)
    
    logits = model(inputs)                           # Forward pass
    loss = cross_entropy(logits.reshape(-1, vocab_size), 
                         targets.reshape(-1))         # Next-token prediction loss
    
    loss.backward()                                   # Backpropagation
    optimizer.step()                                  # Update weights
    optimizer.zero_grad()                             # Reset gradients
```

Each training step: (1) forward pass computes predictions, (2) loss measures how wrong the predictions are, (3) backpropagation computes gradients, (4) optimizer updates weights. Repeat billions of times.

---

## 3. Optimizers: The Engine of Learning

Before we train billions of parameters, we need to understand *how* parameters get updated. Choosing the right optimizer is critical for successful pre-training.

### 3.1 Gradient Descent Fundamentals

At its core, training is an optimization problem. We have a loss function $\mathcal{L}(\theta)$ and we want to find parameters $\theta$ that minimize it. **Gradient descent** is the fundamental algorithm:

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)
$$

Where $\eta$ is the **learning rate** (step size) and $\nabla \mathcal{L}$ is the **gradient** (direction of steepest ascent). We subtract because we want to go *downhill*.

**Intuition**: Imagine standing on a hilly landscape in thick fog. You can only feel the slope beneath your feet. Gradient descent says: "take a step in the direction that goes most steeply downhill."

### 3.2 SGD: Stochastic Gradient Descent

Computing the exact gradient requires processing the entire dataset — impractical for billions of tokens. **Stochastic Gradient Descent (SGD)** approximates it using mini-batches:

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}_{\text{batch}}(\theta_t)
$$

**Pros**: Fast, memory-efficient
**Cons**: Noisy gradients, can oscillate, sensitive to learning rate

### 3.3 Momentum: Adding Inertia

Plain SGD can oscillate in "ravines" — narrow valleys where the gradient bounces back and forth. **Momentum** adds a velocity term that accumulates past gradients:

$$
v_t = \beta v_{t-1} + \nabla \mathcal{L}(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - \eta v_t
$$

Where $\beta \approx 0.9$ is the momentum coefficient.

**Intuition**: Like a ball rolling downhill — it builds up speed and can coast through small bumps. The accumulated velocity helps smooth out noisy gradients, accelerate through consistent directions, and escape shallow local minima.

### 3.4 Adam: The King of LLM Training

**Adam** (Adaptive Moment Estimation) combines momentum with per-parameter learning rates. It maintains two exponential moving averages:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment: like momentum)}
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment: gradient variance)}
$$

The parameter update:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Where $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected versions.

**Why Adam dominates LLM training:**

| Feature | Benefit for LLMs |
|---------|------------------|
| **Adaptive learning rates** | Different layers can learn at different speeds |
| **Momentum** | Stable training despite noisy gradients |
| **Robust to hyperparameters** | Works well with default settings |
| **Handles sparse gradients** | Important for embedding layers |

**Typical settings for LLM pre-training:**
- $\beta_1 = 0.9$ (momentum decay)
- $\beta_2 = 0.95$ or $0.999$ (variance decay)
- $\epsilon = 10^{-8}$ (numerical stability)
- Learning rate: 1e-4 to 6e-4 with warmup + cosine decay

### 3.5 Why C ≈ 6ND?

In the [Scaling Laws](day09-scaling-laws.md) article, we mentioned compute $C \approx 6ND$. Now we can understand why:

**Factor of 2**: Forward pass + backward pass (each touches all $N$ parameters)

**Factor of 3**: Adam maintains 3 quantities per parameter:
1. The gradient $g$
2. First moment $m$
3. Second moment $v$

Hence: $C \approx 2 \times 3 \times N \times D = 6ND$

| Optimizer | Approximate C | Notes |
|-----------|---------------|-------|
| SGD | ~2ND | No extra state |
| SGD + Momentum | ~4ND | One moment term |
| **Adam** | **~6ND** | Two moment terms |

### 3.6 Modern Variants

| Optimizer | Key Idea | When to use |
|-----------|----------|-------------|
| **AdamW** | Decoupled weight decay | Standard for LLMs |
| **AdaFactor** | Memory-efficient Adam | When memory is tight |
| **LAMB** | Layer-wise adaptive rates | Very large batch training |
| **Lion** | Sign-based updates | Emerging alternative |
| **Muon** | Orthogonalized updates | Cutting-edge, used in NanoGPT speedruns |

---

## 4. Learning Rate Scheduling

The learning rate is perhaps the most important hyperparameter. Too high: training diverges. Too low: training is slow and may get stuck.

### 4.1 The Standard Recipe: Warmup + Cosine Decay

```
Learning Rate
    ↑
    |    /‾‾‾‾‾‾‾‾‾‾‾\
    |   /              \
    |  /                \____
    | /                       \___
    |/                             \___
    +--------------------------------→ Steps
      Warmup    Peak      Cosine Decay
```

**Warmup (first ~1–5% of training)**:
- Start with a tiny learning rate, linearly increase to peak
- Prevents early instability when the model is randomly initialized

**Peak (brief plateau)**:
- Maximum learning rate
- Where most learning happens

**Cosine decay (rest of training)**:
- Gradually reduce learning rate following a cosine curve to near-zero

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\!\left(1 + \cos\!\left(\frac{\pi \cdot t}{T}\right)\right)
$$

Where $\eta_t$ is the learning rate at step $t$, $T$ is total steps, and $\eta_{\max}$ / $\eta_{\min}$ are the peak and minimum learning rates. Getting this schedule wrong can waste millions of dollars in compute.

**Why cosine decay?** Empirically, it outperforms linear and step decay. The smooth decay matches the natural optimization landscape, and unlike step decay, it never completely stops learning.

---

## 5. The Data — What Do We Feed These Models?

Pre-training data quality and diversity directly determine model capability. The old adage "garbage in, garbage out" applies here with full force.

![Figure 2: Typical pre-training data mixture and the training pipeline](../zh/images/day11/loss-curve-and-data-mix.png)
*Figure 2: Left — A typical pre-training loss curve showing rapid early learning and slow refinement. Right — Illustrative data mixture proportions.*

### 5.1 Data Sources

Modern pre-training corpora draw from diverse sources:

| Source | Proportion | What It Provides |
|--------|-----------|-----------------|
| **Web crawl** (CommonCrawl) | 40–60% | Breadth, general knowledge, multilingual text |
| **Code repositories** (GitHub) | 15–25% | Logical reasoning, structured thinking |
| **Books** | 10–15% | Long-form reasoning, coherent narratives |
| **Scientific papers** (arXiv) | 5–10% | Technical knowledge, mathematical reasoning |
| **Wikipedia** | 3–5% | High-quality factual knowledge |
| **Curated datasets** | 5–10% | Specific domains (math, legal, medical) |

### 5.2 Data Cleaning — The Unsung Hero

Raw CommonCrawl data is messy. A rigorous pipeline is essential:

| Stage | What happens | Why it matters |
|-------|--------------|----------------|
| **Cleaning** | Remove HTML, fix encoding | Garbage in, garbage out |
| **Deduplication** | Remove exact and fuzzy duplicates at document/paragraph level | Prevents memorization, saves compute |
| **Filtering** | Quality scoring, language detection | Quality > quantity |
| **PII removal** | Scrub personally identifiable information | Privacy |
| **Toxic content filtering** | Remove harmful content (partially — debated) | Safety |

> **Why deduplication matters**: If the same text appears 1,000 times, the model wastes capacity memorizing it instead of learning generalizable patterns. Llama 3's technical report showed deduplication significantly improves downstream performance.

### 5.3 Data Scale

The trend has been staggering:

- **GPT-2 (2019)**: ~40 GB of text (~10B tokens)
- **GPT-3 (2020)**: ~570 GB (~300B tokens)
- **Llama 2 (2023)**: ~2T tokens
- **Llama 3 (2024)**: ~15T tokens
- **Modern estimates**: Some models reportedly trained on 15–20T+ tokens

The Chinchilla scaling laws (Day 9) suggest that for a given model size, there's an optimal amount of training data. Many early models were *under-trained* relative to their parameter count.

---

## 6. Compute Requirements — Why This Costs Millions

### 6.1 The Scale

Training a frontier LLM requires enormous compute:

| Model | Parameters | GPUs | Training Duration | Estimated Cost |
|-------|-----------|------|-------------------|----------------|
| GPT-3 | 175B | ~10,000 V100s | ~1 month | ~$5M |
| Llama 2 70B | 70B | ~2,000 A100s | ~2 months | ~$3M |
| Llama 3 405B | 405B | ~16,000 H100s | ~1–2 months | ~$30–50M |
| Frontier (est.) | 1T+ | 50,000+ H100s | Months | $100M+ |

### 6.2 Distributed Training: 3D Parallelism

Training at this scale requires combining multiple parallelism strategies:

**Data Parallelism** — Split the batch across GPUs. Each GPU computes gradients independently, then synchronize (all-reduce):

```
        Global Batch (e.g., 4M tokens)
        ↓
┌───────┼───────┼───────┼───────┐
│GPU 0  │GPU 1  │GPU 2  │GPU 3  │  (1M tokens each)
│compute│compute│compute│compute│
│grads  │grads  │grads  │grads  │
└───────┴───────┴───────┴───────┘
        ↓ all-reduce
     Average gradients → Update parameters (synchronized)
```

**Tensor Parallelism** — Split individual matrix operations across GPUs. A single attention layer's computation spans multiple GPUs:

```
Single Transformer Layer:
┌─────────────────────────────────────┐
│  Attention (split across GPUs)      │
│  GPU0: heads 0-3 | GPU1: heads 4-7  │
├─────────────────────────────────────┤
│  MLP (split across GPUs)            │
│  GPU0: first half | GPU1: second    │
└─────────────────────────────────────┘
```

**Pipeline Parallelism** — Different transformer layers live on different GPUs. Data flows through the pipeline like an assembly line:

```
Time →
GPU 0 (layers 1-8):   [B1][B2][B3][B4]
GPU 1 (layers 9-16):     [B1][B2][B3][B4]
GPU 2 (layers 17-24):       [B1][B2][B3][B4]
GPU 3 (layers 25-32):          [B1][B2][B3][B4]
```

**The full picture:**

| Dimension | What it splits | Typical scale |
|-----------|----------------|---------------|
| Data parallel | Batches | 8–64 replicas |
| Tensor parallel | Within layers | 4–8 GPUs |
| Pipeline parallel | Across layers | 4–16 stages |

Example: Training a 175B parameter model might use 8-way tensor parallelism × 8-way pipeline parallelism × 64-way data parallelism = **4,096 GPUs**.

### 6.3 Key Optimizations

**Mixed Precision Training (BF16/FP8)** — Using lower precision floats cuts memory and compute in half with minimal quality loss.

**Gradient Accumulation** — Simulate larger batch sizes by accumulating gradients over multiple forward passes before updating.

**Memory reality**: A 70B parameter model in BF16 requires ~140 GB just to store weights. Add optimizer states (Adam uses 2 extra copies), gradients, and activations — you need ~1–2 TB of GPU memory for training. Model parallelism isn't optional; it's essential.

---

## 7. Training Stability and Debugging

### 7.1 What the Loss Curve Tells Us

![Figure 3: Training timeline and compute allocation — pre-training dominates](../zh/images/day11/compute-allocation.png)
*Figure 3: Pre-training consumes 90%+ of the total compute budget. The entire pipeline from data collection to post-training takes months.*

The loss curve follows a characteristic pattern:

1. **Steep descent** (early training): The model rapidly learns syntax, common word combinations, basic facts
2. **Gradual refinement** (mid training): World knowledge, reasoning patterns, domain-specific knowledge
3. **Asymptotic plateau** (late training): Diminishing returns; the model approaches its irreducible loss (entropy of natural language)

The irreducible loss — the floor the curve approaches — represents the inherent unpredictability of language. Even a perfect model can't predict the next word with certainty, because natural language has genuine ambiguity.

### 7.2 Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss explodes (NaN/Inf) | Learning rate too high | Reduce LR, add gradient clipping |
| Loss plateaus early | LR too low or dead neurons | Increase LR, check activations |
| Loss spikes | Bad data or numerical issues | Filter data, use mixed precision carefully |
| Slow convergence | Poor initialization | Use proper init (e.g., small init for residual branches) |

### 7.3 Essential Monitoring

```python
# Track these metrics during training
- train_loss      # Should decrease smoothly
- grad_norm       # Spikes indicate instability
- learning_rate   # Verify schedule is correct
- throughput      # Tokens/second, detect slowdowns
- memory_usage    # Prevent OOM
```

### 7.4 Gradient Clipping

Prevents exploding gradients by capping the gradient norm:

$$
\hat{g} = \begin{cases} g & \text{if } \|g\| \leq c \\[4pt] c \cdot \dfrac{g}{\|g\|} & \text{otherwise} \end{cases}
$$

Typical value: $c = 1.0$

---

## 8. What Does Pre-training Actually Teach?

This is the most fascinating question. The model is *never explicitly told* to learn grammar, facts, or reasoning. It just predicts the next token. Yet it learns:

### 8.1 Surface Patterns → Deep Understanding

| Training Phase | What Emerges | Example |
|---------------|-------------|---------|
| First 1% of data | Token co-occurrence, basic syntax | "the" → "cat" (common pair) |
| First 10% | Grammar, sentence structure | Produces grammatically correct sentences |
| First 50% | World facts, common knowledge | "Paris is the capital of ___" → "France" |
| 50–100% | Reasoning patterns, style mimicry | Can follow multi-step reasoning chains |

### 8.2 Code Data and Reasoning

One of the most impactful discoveries: **including code in pre-training data significantly improves reasoning ability**, not just coding ability. The structured, logical nature of code appears to teach the model general reasoning patterns that transfer to non-code tasks.

This is why modern pre-training datasets include 15–25% code — not just to make the model code, but to make it *think better*.

### 8.3 What Pre-training Does NOT Teach

Pre-training alone produces a **base model** — powerful but raw:

- ❌ It doesn't follow instructions well ("Summarize this" → continues the text instead)
- ❌ It doesn't know when to refuse harmful requests
- ❌ It doesn't format responses as helpful answers
- ❌ It can complete text but doesn't "chat"

This is why **post-training alignment** (RLHF, instruction tuning) is essential — it transforms a text-completer into a helpful assistant. We'll cover this in Days 13–15.

---

## 9. Modern Trends in Pre-training

### 9.1 Data Starvation

We may be running out of high-quality text. The total amount of human-generated text on the internet is finite (~10–100T tokens of useful content). Some estimates suggest we could exhaust high-quality pre-training data by 2026–2028.

Solutions being explored:
- **Synthetic data**: Using existing models to generate training data
- **Multimodal data**: Training on images, video, audio (vastly more data)
- **Data recycling**: Training on the same data more efficiently
- **Curriculum learning**: Carefully ordering training examples

### 9.2 Multi-stage Pre-training

Modern models often use a multi-stage approach:

1. **Base pre-training**: Massive diverse data, standard learning rate
2. **Annealing**: Gradually reduce learning rate on high-quality curated data
3. **Domain upweighting**: Increase proportion of math/code/data near the end

Llama 3 reportedly used this approach, with the final annealing phase on carefully curated high-quality data significantly boosting performance.

### 9.3 Longer Training on Smaller Models

The Chinchilla insight continues to influence: rather than building bigger models, train smaller models longer. A 7B model trained on 15T tokens can be surprisingly competitive with a 70B model trained on 1T tokens — and is much cheaper to deploy.

---

## 10. Common Misconceptions

### ❌ "Pre-training just memorizes the internet"

While models do memorize some training data (especially rare, repeated text), the vast majority of their capability comes from *generalization*. A model trained on 15T tokens with ~100B parameters physically cannot store even a fraction of the training data. It must learn compressed representations — genuine understanding of patterns.

### ❌ "More data always helps"

Quality matters enormously. Adding low-quality data can actually *hurt* performance by diluting the signal. Llama 3's aggressive data cleaning was reportedly more impactful than adding more raw data.

### ❌ "Pre-training = the model is done"

Pre-training produces a base model — a powerful but raw text completer. Without alignment (instruction tuning, RLHF), it's like a brilliant person who doesn't understand conversation norms. Base models complete text; aligned models *assist*.

---

## 11. Complete Code Example — Minimal Pre-training Loop

```python
import torch
import torch.nn as nn

class SimpleLM(nn.Module):
    """A minimal language model for demonstration."""
    def __init__(self, vocab_size=32000, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads,
            num_encoder_layers=n_layers, batch_first=True
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        out = self.transformer(emb, mask=mask.to(x.device))
        return self.head(out)

# --- Pre-training loop ---
model = SimpleLM().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for step, batch in enumerate(dataloader):
    inputs, targets = batch[:, :-1].cuda(), batch[:, 1:].cuda()

    logits = model(inputs)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 1000 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
```

### A Typical Pre-training Recipe (YAML)

```yaml
# Model
architecture: decoder-only transformer
parameters: 7B
context_length: 4096

# Data
dataset: deduplicated web text
tokens: 1-2T tokens (Chinchilla optimal: ~140B for 7B model)
tokenizer: BPE, 32K vocabulary

# Optimizer
optimizer: AdamW
learning_rate: 3e-4 peak
beta1: 0.9
beta2: 0.95
weight_decay: 0.1
warmup_steps: 2000
schedule: cosine decay to 3e-5

# Training
batch_size: 4M tokens
gradient_clipping: 1.0
precision: bfloat16
parallelism: FSDP or 3D parallel
```

---

## 12. Further Reading

### Beginner
1. [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Visual guide to how autoregressive LMs work
2. [Andrej Karpathy's "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Step-by-step video tutorial

### Advanced
1. [Llama 3 Technical Report](https://ai.meta.com/blog/meta-llama-3/) — State-of-the-art pre-training details
2. [Chinchilla paper (Training Compute-Optimal LLMs)](https://arxiv.org/abs/2203.15556) — Why data scale matters as much as model size
3. [Megatron-LM](https://arxiv.org/abs/1909.08053) — Training multi-billion parameter models using model parallelism

### Papers
1. [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Demonstrated pre-training at scale
2. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — The optimizer behind it all
3. [Data-centric LLM research survey](https://arxiv.org/abs/2402.00111) — How data quality shapes model quality

---

## Reflection Questions

1. If pre-training only teaches next-token prediction, *why* does the model learn reasoning instead of just surface statistics? What is it about language that makes this possible?
2. We may run out of human-generated training data. What are the implications of training on synthetic data generated by other models? Could this create an "inbreeding" problem?
3. The Chinchilla laws suggest we should train smaller models longer. What are the practical trade-offs between a 70B model trained optimally vs. a 405B model trained sub-optimally?
4. Adam uses ~3× more memory per parameter than plain SGD. When might you prefer a simpler optimizer despite the theoretical advantages of Adam?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| Pre-training | Self-supervised learning on massive unlabeled text to build general language understanding |
| Self-supervised learning | Labels come from the data itself (e.g., next token), no human annotation needed |
| Causal LM | Predict next token given past context — the dominant objective for modern LLMs |
| Cross-entropy loss | Measures how "surprised" the model is by the actual next token |
| Adam/AdamW | Adaptive optimizer with momentum — standard for LLM training, explains the C ≈ 6ND compute formula |
| LR schedule | Warmup + cosine decay prevents instability and enables fine-grained optimization |
| 3D parallelism | Data + tensor + pipeline parallelism to distribute training across thousands of GPUs |
| Data mixture | Diverse sources (web, code, books, papers) with careful cleaning and deduplication |
| Base model | The raw output of pre-training — powerful but unaligned, needs post-training |
| Data starvation | Concern that we may exhaust high-quality human-generated text in coming years |

**Key Takeaway**: Pre-training is the foundation of every LLM — a massive self-supervised learning process that turns random parameters into a model with broad language understanding. The objective is simple (predict the next token), the optimizer is Adam with a warmup + cosine decay schedule, and the engineering is heroic (3D parallelism across thousands of GPUs). It works because predicting the next token in natural text *forces* the model to build internal representations of grammar, facts, and reasoning. But it's only the first step: the base model must then be aligned (post-training) to become a useful assistant.

---

*Day 11 of 60 | LLM Fundamentals*
*Next: [Day 12 — Fine-tuning: Adapting Pre-trained Models](day12-finetuning.md)*
