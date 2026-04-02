# Day 11: Pre-training — Teaching Machines to Read the Internet

> **Core Question**: How do we train a model on trillions of tokens to become a capable language model?

---

## Opening

Imagine teaching someone a new language by having them read *every book ever written*—without translations, without teachers, just pure immersion. That's essentially what pre-training does for language models: we expose them to massive amounts of text and let them figure out the patterns on their own.

Pre-training is the foundation of modern LLMs. It's where models develop their "world knowledge," learning everything from grammar and facts to reasoning patterns and coding skills. This single training phase can cost tens of millions of dollars and take months on thousands of GPUs.

Today we'll explore how this magic works—starting with the surprisingly simple training objective, through the critical choice of optimizer, to the engineering challenges of training at scale.

---

## 1. The Training Objective: Next-Token Prediction

### 1.1 Autoregressive Language Modeling

The core idea is remarkably simple: **predict the next word**.

Given a sequence of tokens $x_1, x_2, ..., x_{t-1}$, the model learns to predict $x_t$:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, x_2, ..., x_{t-1})
$$

This is called **cross-entropy loss**—the model outputs a probability distribution over all possible next tokens, and we penalize it for assigning low probability to the actual next token.

### 1.2 Why This Works So Well

The next-token prediction objective implicitly teaches the model:

| What it learns | How it's learned |
|----------------|------------------|
| **Grammar** | Can't predict "She go to store" correctly |
| **Facts** | "The capital of France is ___" requires knowing Paris |
| **Reasoning** | "2 + 2 = ___" requires arithmetic |
| **Common sense** | "The ice cream melted because it was ___" requires physics |
| **Style** | Different contexts need different completions |

The beauty is that we don't need labeled data—the text itself provides the supervision. This is **self-supervised learning**.

### 1.3 The Training Data Pipeline

```
Raw Internet Text → Cleaning → Deduplication → Filtering → Tokenization → Shards
```

| Stage | What happens | Why it matters |
|-------|--------------|----------------|
| **Cleaning** | Remove HTML, fix encoding | Garbage in, garbage out |
| **Deduplication** | Remove repeated content | Prevents memorization, saves compute |
| **Filtering** | Quality scoring, language detection | Quality > quantity |
| **Tokenization** | Convert text to token IDs | Determines vocabulary size |
| **Sharding** | Split into files for distributed loading | Enables parallel training |

---

## 2. Optimizers: The Engine of Learning

Before we train billions of parameters, we need to understand *how* parameters get updated. This is where optimizers come in—and choosing the right one is critical for successful pre-training.

### 2.1 The Basics: Gradient Descent

At its core, training is an optimization problem. We have a loss function $\mathcal{L}(\theta)$ that measures how bad our predictions are, and we want to find parameters $\theta$ that minimize it.

**Gradient descent** is the fundamental algorithm:

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)
$$

Where:
- $\eta$ is the **learning rate** (step size)
- $\nabla \mathcal{L}$ is the **gradient** (direction of steepest ascent)
- We subtract because we want to go *downhill*

**Intuition**: Imagine standing on a hilly landscape in thick fog. You can only feel the slope beneath your feet. Gradient descent says: "take a step in the direction that goes most steeply downhill."

### 2.2 SGD: Stochastic Gradient Descent

Computing the exact gradient requires processing the entire dataset—impractical for billions of tokens. **Stochastic Gradient Descent (SGD)** approximates it using mini-batches:

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}_{\text{batch}}(\theta_t)
$$

**Pros**: Fast, memory-efficient
**Cons**: Noisy gradients, can oscillate, sensitive to learning rate

### 2.3 Momentum: Adding Inertia

Plain SGD can oscillate in "ravines"—narrow valleys where the gradient bounces back and forth. **Momentum** adds a velocity term that accumulates past gradients:

$$
v_t = \beta v_{t-1} + \nabla \mathcal{L}(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - \eta v_t
$$

Where $\beta \approx 0.9$ is the momentum coefficient.

**Intuition**: Like a ball rolling downhill—it builds up speed and can coast through small bumps. The accumulated velocity helps:
- Smooth out noisy gradients
- Accelerate through consistent directions
- Escape shallow local minima

### 2.4 Adam: The King of LLM Training

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

### 2.5 Why C ≈ 6ND?

In the [Scaling Laws](day09-scaling-laws.md) article, we mentioned compute $C \approx 6ND$. Now we can understand why:

**Factor of 2**: Forward pass + backward pass (each touches all N parameters)

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

### 2.6 Modern Variants

| Optimizer | Key Idea | When to use |
|-----------|----------|-------------|
| **AdamW** | Decoupled weight decay | Standard for LLMs |
| **AdaFactor** | Memory-efficient Adam | When memory is tight |
| **LAMB** | Layer-wise adaptive rates | Very large batch training |
| **Lion** | Sign-based updates | Emerging alternative |
| **Muon** | Orthogonalized updates | Cutting-edge, used in NanoGPT speedruns |

---

## 3. Learning Rate Schedules

The learning rate is perhaps the most important hyperparameter. Too high: training diverges. Too low: training is slow and may get stuck.

### 3.1 The Standard Recipe: Warmup + Decay

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

**Warmup (first ~1-5% of training)**:
- Start with tiny learning rate
- Linearly increase to peak
- Prevents early instability when model is randomly initialized

**Peak (brief plateau)**:
- Maximum learning rate
- Where most learning happens

**Decay (rest of training)**:
- Gradually reduce learning rate
- Cosine schedule is most popular: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$
- Allows fine-grained optimization at the end

### 3.2 Why Cosine Decay?

Empirically, cosine decay outperforms linear decay and step decay for LLM training. Hypotheses:
- Smooth decay matches the natural optimization landscape
- Never completely stops learning (unlike step decay)
- Elegant mathematical properties

---

## 4. Training at Scale

### 4.1 Data Parallelism

The simplest form of distributed training: split the batch across GPUs.

```
        Global Batch (e.g., 4M tokens)
        ↓
┌───────┼───────┼───────┼───────┐
│GPU 0  │GPU 1  │GPU 2  │GPU 3  │  (1M tokens each)
│       │       │       │       │
│compute│compute│compute│compute│
│grads  │grads  │grads  │grads  │
└───────┴───────┴───────┴───────┘
        ↓ all-reduce
     Average gradients
        ↓
   Update parameters (synchronized)
```

### 4.2 Tensor Parallelism

For models too large to fit on one GPU: split the model's layers across GPUs.

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

### 4.3 Pipeline Parallelism

Split layers across GPUs, process micro-batches in a pipeline:

```
Time →
GPU 0 (layers 1-8):   [B1][B2][B3][B4]
GPU 1 (layers 9-16):     [B1][B2][B3][B4]
GPU 2 (layers 17-24):       [B1][B2][B3][B4]
GPU 3 (layers 25-32):          [B1][B2][B3][B4]
```

### 4.4 The Full Picture: 3D Parallelism

Modern large-scale training combines all three:

| Dimension | What it splits | Typical scale |
|-----------|----------------|---------------|
| Data parallel | Batches | 8-64 replicas |
| Tensor parallel | Within layers | 4-8 GPUs |
| Pipeline parallel | Across layers | 4-16 stages |

Example: Training a 175B parameter model might use:
- 8-way tensor parallelism
- 8-way pipeline parallelism  
- 64-way data parallelism
- Total: 8 × 8 × 64 = 4,096 GPUs

---

## 5. Training Stability and Debugging

### 5.1 Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss explodes (NaN/Inf) | Learning rate too high | Reduce LR, add gradient clipping |
| Loss plateaus early | LR too low or dead neurons | Increase LR, check activations |
| Loss spikes | Bad data or numerical issues | Filter data, use mixed precision carefully |
| Slow convergence | Poor initialization | Use proper init (e.g., small init for residual branches) |

### 5.2 Essential Monitoring

```python
# Track these metrics
- train_loss      # Should decrease smoothly
- grad_norm       # Spikes indicate instability  
- learning_rate   # Verify schedule is correct
- throughput      # Tokens/second, detect slowdowns
- memory_usage    # Prevent OOM
```

### 5.3 Gradient Clipping

Prevents exploding gradients by capping the gradient norm:

$$
\hat{g} = \begin{cases} g & \text{if } ||g|| \leq c \\ c \cdot \frac{g}{||g||} & \text{otherwise} \end{cases}
$$

Typical value: $c = 1.0$

---

## 6. Putting It All Together

### 6.1 A Typical Pre-training Recipe

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

### 6.2 Cost Estimates

| Model Size | Tokens | GPUs | Time | Cost |
|------------|--------|------|------|------|
| 7B | 1T | 256 A100 | ~1 week | ~$100K |
| 70B | 2T | 2000 A100 | ~2 weeks | ~$2M |
| 175B | 300B | 1000 V100 | ~1 month | ~$5M* |
| 1T+ | 2T+ | 10000+ H100 | months | ~$100M+ |

*GPT-3 was undertrained by modern standards

---

## 7. Key Takeaways

1. **Pre-training objective is simple**: Just predict the next token. The magic is in scale.

2. **Optimizer choice matters**: Adam (and AdamW) dominate because they're robust and adaptive. The C ≈ 6ND formula assumes Adam's two moment terms.

3. **Learning rate schedule is critical**: Warmup prevents early instability, cosine decay enables fine-grained optimization.

4. **Scaling requires all three forms of parallelism**: Data, tensor, and pipeline parallelism each solve different bottlenecks.

5. **Stability is hard-won**: Gradient clipping, careful initialization, and constant monitoring are essential at scale.

---

## Further Reading

- **Adam paper**: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- **Training GPT-3**: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- **LLaMA training**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **Megatron-LM**: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

---

*Next: [Day 12 — Fine-tuning: Adapting Pre-trained Models](day12-finetuning.md)*
