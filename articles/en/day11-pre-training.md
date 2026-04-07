# Day 11: Pre-training — How LLMs Learn to Model the World

> **Core Question**: How does a model go from random noise to understanding language, just by reading the internet?

---

## Opening

Imagine you're dropped into a massive library with billions of books, and your only task is: *predict the next word on every page*. No teacher, no labels, no instructions — just predict, predict, predict. After reading enough pages, something remarkable happens: you start to understand grammar, facts, reasoning patterns, even humor.

This is essentially what pre-training does. It's the single most expensive, most important phase of building an LLM — the process that turns random parameter initialization into something that genuinely understands language. GPT-4, Llama, Gemini — every major LLM went through this phase, consuming trillions of tokens on thousands of GPUs for months.

Think of pre-training like raising a child through sheer exposure to books. You don't sit down and explicitly teach grammar rules or geography facts. Instead, the child reads millions of pages and *absorbs* the patterns. Pre-training is the "reading millions of pages" part. The alignment phase (which we'll cover in Day 13-15) is the "teaching manners" part.

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
- $P(x_t | x_1, x_2, ..., x_{t-1})$
- Autoregressive: can only see the past, not the future
- Optimized for text *generation*

**Masked Language Modeling (MLM)** — used by BERT:
- Randomly mask ~15% of tokens, predict them from context (both directions)
- $P(x_{\text{masked}} | x_{\text{surrounding}})$
- Bidirectional: sees both left and right context
- Optimized for text *understanding*

**Prefix Language Modeling (Prefix LM)** — used by T5:
- Split input into prefix (visible) and target (predict)
- Like CLM but with a bidirectional prefix section
- Unifies understanding and generation

Modern LLMs (GPT-4, Llama, etc.) almost exclusively use **Causal LM** because it naturally produces generation-capable models.

---

## 2. The Data — What Do We Feed These Models?

Pre-training data quality and diversity directly determine model capability. The old adage "garbage in, garbage out" applies here with full force.

![Figure 2: Typical pre-training data mixture and the training pipeline](../zh/images/day11/loss-curve-and-data-mix.png)
*Figure 2: Left — A typical pre-training loss curve showing rapid early learning and slow refinement. Right — Illustrative data mixture proportions.*

### 2.1 Data Sources

Modern pre-training corpora draw from diverse sources:

| Source | Proportion | What It Provides |
|--------|-----------|-----------------|
| **Web crawl** (CommonCrawl) | 40-60% | Breadth, general knowledge, multilingual text |
| **Code repositories** (GitHub) | 15-25% | Logical reasoning, structured thinking |
| **Books** | 10-15% | Long-form reasoning, coherent narratives |
| **Scientific papers** (arXiv) | 5-10% | Technical knowledge, mathematical reasoning |
| **Wikipedia** | 3-5% | High-quality factual knowledge |
| **Curated datasets** | 5-10% | Specific domains (math, legal, medical) |

### 2.2 Data Cleaning — The Unsung Hero

Raw CommonCrawl data is messy. A rigorous pipeline is essential:

1. **Language filtering** — Remove undesired languages or low-quality text
2. **Deduplication** — Exact and fuzzy deduplication at document and paragraph level (critical to prevent memorization)
3. **Quality filtering** — Classifier-based filtering to remove spam, low-quality content
4. **PII removal** — Scrub personally identifiable information
5. **Toxic content filtering** — Remove harmful content (partially — this is debated)

> **Why deduplication matters**: If the same text appears 1,000 times, the model wastes capacity memorizing it instead of learning generalizable patterns. Llama 3's technical report showed deduplication significantly improves downstream performance.

### 2.3 Data Scale

The trend has been staggering:

- **GPT-2 (2019)**: ~40 GB of text (~10B tokens)
- **GPT-3 (2020)**: ~570 GB (~300B tokens)
- **Llama 2 (2023)**: ~2T tokens
- **Llama 3 (2024)**: ~15T tokens
- **Modern estimates**: Some models reportedly trained on 15-20T+ tokens

The Chinchilla scaling laws (Day 9) suggest that for a given model size, there's an optimal amount of training data. Many early models were *under-trained* relative to their parameter count.

---

## 3. How Training Actually Works

### 3.1 The Training Loop

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

### 3.2 The Loss Function

The pre-training objective is **cross-entropy loss** (equivalent to negative log-likelihood):

$$
\begin{aligned}
\mathcal{L} &= -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta) \\
&= -\frac{1}{T} \sum_{t=1}^{T} \log \text{softmax}(z_t)[x_t]
\end{aligned}
$$

Where $\theta$ are the model parameters, $z_t$ is the logit vector at position $t$, and $x_t$ is the actual next token. Lower loss means better prediction — the model is more "surprised" less often.

### 3.3 What the Loss Curve Tells Us

![Figure 3: Training timeline and compute allocation — pre-training dominates](../zh/images/day11/compute-allocation.png)
*Figure 3: Pre-training consumes 90%+ of the total compute budget. The entire pipeline from data collection to post-training takes months.*

The loss curve follows a characteristic pattern:

1. **Steep descent** (early training): The model rapidly learns syntax, common word combinations, basic facts
2. **Gradual refinement** (mid training): World knowledge, reasoning patterns, domain-specific knowledge
3. **Asymptotic plateau** (late training): Diminishing returns; the model approaches its irreducible loss (entropy of natural language)

The irreducible loss — the floor the curve approaches — represents the inherent unpredictability of language. Even a perfect model can't predict the next word with certainty, because natural language has genuine ambiguity.

---

## 4. Compute Requirements — Why This Costs Millions

### 4.1 The Scale

Training a frontier LLM requires enormous compute:

| Model | Parameters | GPUs | Training Duration | Estimated Cost |
|-------|-----------|------|-------------------|----------------|
| GPT-3 | 175B | ~10,000 V100s | ~1 month | ~$5M |
| Llama 2 70B | 70B | ~2,000 A100s | ~2 months | ~$3M |
| Llama 3 405B | 405B | ~16,000 H100s | ~1-2 months | ~$30-50M |
| Frontier (est.) | 1T+ | 50,000+ H100s | Months | $100M+ |

### 4.2 Key Optimizations

Training at this scale requires careful engineering:

**Mixed Precision Training (BF16/FP8)** — Using lower precision (16-bit or 8-bit floats instead of 32-bit) cuts memory and compute in half with minimal quality loss.

**Tensor Parallelism** — Split individual matrix operations across GPUs. A single attention layer's computation spans multiple GPUs.

**Pipeline Parallelism** — Different transformer layers live on different GPUs. Data flows through the pipeline like an assembly line.

**Data Parallelism** — Split the batch across GPU groups. Each group computes gradients independently, then synchronize (all-reduce).

**Gradient Accumulation** — Simulate larger batch sizes by accumulating gradients over multiple forward passes before updating.

### 4.3 Learning Rate Scheduling

Pre-training uses a carefully tuned learning rate schedule, not a fixed rate:

1. **Warmup** (first 0.1-2% of steps): Linearly increase LR from near-zero to peak (e.g., 3e-4). This prevents early instability where large gradients could destabilize random initialization.

2. **Stable phase**: Maintain peak LR for most of training.

3. **Cosine decay / Annealing**: Gradually reduce LR following a cosine curve to near-zero. This allows the model to settle into a sharp minimum.

$$
\begin{aligned}
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)
\end{aligned}
$$

Where $\eta_t$ is the learning rate at step $t$, $T$ is total steps, and $\eta_{\max}$ / $\eta_{\min}$ are the peak and minimum learning rates. Getting this schedule wrong can waste millions of dollars in compute.

### 4.4 Memory vs. Compute

A 70B parameter model in BF16 requires ~140 GB just to store weights. Add optimizer states (Adam uses 2 extra copies), gradients, and activations — you need ~1-2 TB of GPU memory for training. This is why model parallelism isn't optional; it's essential.

---

## 5. What Does Pre-training Actually Teach?

This is the most fascinating question. The model is *never explicitly told* to learn grammar, facts, or reasoning. It just predicts the next token. Yet it learns:

### 5.1 Surface Patterns → Deep Understanding

| Training Phase | What Emerges | Example |
|---------------|-------------|---------|
| First 1% of data | Token co-occurrence, basic syntax | "the" → "cat" (common pair) |
| First 10% | Grammar, sentence structure | Produces grammatically correct sentences |
| First 50% | World facts, common knowledge | "Paris is the capital of ___" → "France" |
| 50-100% | Reasoning patterns, style mimicry | Can follow multi-step reasoning chains |

### 5.2 Code Data and Reasoning

One of the most impactful discoveries: **including code in pre-training data significantly improves reasoning ability**, not just coding ability. The structured, logical nature of code appears to teach the model general reasoning patterns that transfer to non-code tasks.

This is why modern pre-training datasets include 15-25% code — not just to make the model code, but to make it *think better*.

### 5.3 What Pre-training Does NOT Teach

Pre-training alone produces a **base model** — powerful but raw:

- ❌ It doesn't follow instructions well ("Summarize this" → continues the text instead)
- ❌ It doesn't know when to refuse harmful requests
- ❌ It doesn't format responses as helpful answers
- ❌ It can complete text but doesn't "chat"

This is why **post-training alignment** (RLHF, instruction tuning) is essential — it transforms a text-completer into a helpful assistant. We'll cover this in Days 13-15.

---

## 6. Modern Trends in Pre-training

### 6.1 Data Starvation

We may be running out of high-quality text. The total amount of human-generated text on the internet is finite (~10-100T tokens of useful content). Some estimates suggest we could exhaust high-quality pre-training data by 2026-2028.

Solutions being explored:
- **Synthetic data**: Using existing models to generate training data
- **Multimodal data**: Training on images, video, audio (vastly more data)
- **Data recycling**: Training on the same data more efficiently
- **Curriculum learning**: Carefully ordering training examples

### 6.2 Multi-stage Pre-training

Modern models often use a multi-stage approach:

1. **Base pre-training**: Massive diverse data, standard learning rate
2. **Annealing**: Gradually reduce learning rate on high-quality curated data
3. **Domain upweighting**: Increase proportion of math/code/data near the end

Llama 3 reportedly used this approach, with the final annealing phase on carefully curated high-quality data significantly boosting performance.

### 6.3 Longer Training on Smaller Models

The Chinchilla insight continues to influence: rather than building bigger models, train smaller models longer. A 7B model trained on 15T tokens can be surprisingly competitive with a 70B model trained on 1T tokens — and is much cheaper to deploy.

---

## 7. Common Misconceptions

### ❌ "Pre-training just memorizes the internet"

While models do memorize some training data (especially rare, repeated text), the vast majority of their capability comes from *generalization*. A model trained on 15T tokens with ~100B parameters physically cannot store even a fraction of the training data. It must learn compressed representations — genuine understanding of patterns.

### ❌ "More data always helps"

Quality matters enormously. Adding low-quality data can actually *hurt* performance by diluting the signal. Llama 3's aggressive data cleaning was reportedly more impactful than adding more raw data.

### ❌ "Pre-training = the model is done"

Pre-training produces a base model — a powerful but raw text completer. Without alignment (instruction tuning, RLHF), it's like a brilliant person who doesn't understand conversation norms. Base models complete text; aligned models *assist*.

---

## 8. Code Example — Minimal Pre-training Loop

```python
import torch
import torch.nn as nn

class SimpleLM(nn.Module):
    """A minimal language model for demonstration."""
    def __init__(self, vocab_size=32000, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # In practice: use a full Transformer, not just a placeholder
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads, 
            num_encoder_layers=n_layers, batch_first=True
        )
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_len) token IDs
        emb = self.embedding(x)  # (batch, seq_len, d_model)
        # Causal mask: prevent attending to future tokens
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        out = self.transformer(emb, mask=mask.to(x.device))
        return self.head(out)  # (batch, seq_len, vocab_size)

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

This is a simplified illustration. Real pre-training uses distributed training, mixed precision, gradient checkpointing, and many more optimizations.

---

## 9. Further Reading

### Beginner
1. [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Visual guide to how autoregressive LMs work
2. [Andrej Karpathy's "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Step-by-step video tutorial

### Advanced
1. [Llama 3 Technical Report](https://ai.meta.com/blog/meta-llama-3/) — State-of-the-art pre-training details
2. [Chinchilla paper (Training Compute-Optimal LLMs)](https://arxiv.org/abs/2203.15556) — Why data scale matters as much as model size

### Papers
1. [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Demonstrated pre-training at scale
2. [Data-centric LLM research survey](https://arxiv.org/abs/2402.00111) — How data quality shapes model quality

---

## Reflection Questions

1. If pre-training only teaches next-token prediction, *why* does the model learn reasoning instead of just surface statistics? What is it about language that makes this possible?
2. We may run out of human-generated training data. What are the implications of training on synthetic data generated by other models? Could this create an "inbreeding" problem?
3. The Chinchilla laws suggest we should train smaller models longer. What are the practical trade-offs between a 70B model trained optimally vs. a 405B model trained sub-optimally?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| Pre-training | Self-supervised learning on massive unlabeled text to build general language understanding |
| Self-supervised learning | Labels come from the data itself (e.g., next token), no human annotation needed |
| Causal LM | Predict next token given past context — the dominant objective for modern LLMs |
| Cross-entropy loss | Measures how "surprised" the model is by the actual next token |
| Data mixture | Diverse sources (web, code, books, papers) with careful cleaning and deduplication |
| Base model | The raw output of pre-training — powerful but unaligned, needs post-training |
| Data starvation | Concern that we may exhaust high-quality human-generated text in coming years |

**Key Takeaway**: Pre-training is the foundation of every LLM — a massive self-supervised learning process that turns random parameters into a model with broad language understanding. It works because predicting the next token in natural text requires understanding grammar, facts, and reasoning — the prediction task *forces* the model to build internal representations of the world. But it's only the first step: the base model must then be aligned (post-training) to become a useful assistant.

---

*Day 11 of 60 | LLM Fundamentals*
*Word count: ~2200 | Reading time: ~11 minutes*
