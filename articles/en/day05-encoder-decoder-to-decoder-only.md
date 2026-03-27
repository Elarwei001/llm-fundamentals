# Day 5: Encoder-Decoder to Decoder-only — BERT vs GPT, Why GPT Won

> **Core Question**: Why did decoder-only architectures (GPT) come to dominate modern AI, even though BERT initially seemed like the clear winner?

---

## Opening: The Exam Analogy

Imagine two students taking a fill-in-the-blank exam.

**BERT** reads the entire exam paper first — scanning every question, every surrounding sentence — before filling in any blank. It's like an editor who absorbs the full document and then makes targeted edits. This bidirectional approach makes BERT exceptionally good at *understanding* text.

**GPT** writes an essay from scratch, one word at a time. It never looks ahead — each word is chosen based only on everything written so far. It's like a writer in full flow, generating naturally without revision.

Both approaches seemed equally promising in 2018. Yet by 2022, GPT-style models had completely reshaped the AI landscape, spawning ChatGPT and triggering the current LLM explosion. BERT, despite its initial dominance, became a specialized tool rather than the foundation for the future.

Why? The answer lies in four concrete mechanisms — and understanding them will transform how you think about architecture choices in deep learning.

![Architecture Comparison](../zh/images/day05/architecture-comparison-v3.png)
*Figure 1: Three transformer architectures — Encoder-only (BERT), Decoder-only (GPT), and Encoder-Decoder (T5). Each makes different tradeoffs between understanding and generation.*

---

## 1. The Three Architectures: Layer-by-Layer Deep Dive

The 2017 Transformer paper ([Vaswani et al.](https://arxiv.org/abs/1706.03762)) introduced an encoder-decoder model for machine translation. But researchers quickly realized the architecture could be split, specialized, and scaled in different ways. By 2018–2020, three distinct paradigms had emerged.

Understanding *why* each architecture made specific design choices requires examining each layer's purpose and the trade-offs involved.

### 1.1 Encoder-only: BERT (2018)

BERT ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)) strips the Transformer down to its encoder half. Let's examine each component:

#### Layer Structure (repeated L times, typically L=12 or L=24)

```
Input Embeddings
       ↓
┌─────────────────────────────────────┐
│  Multi-Head Bidirectional Attention │  ← Every token sees every other token
│  + Residual Connection + LayerNorm  │
├─────────────────────────────────────┤
│  Feed-Forward Network (FFN)         │  ← Non-linear transformation
│  + Residual Connection + LayerNorm  │
└─────────────────────────────────────┘
       ↓ (repeat L times)
Final Hidden States → Task-specific head
```

#### Component 1: Input Embeddings

**What it does**: Converts tokens to dense vectors and adds positional information.

```
Token Embedding + Segment Embedding + Position Embedding = Input
     (30522 × 768)      (2 × 768)         (512 × 768)
```

**Why this design**:
- **Token Embedding**: Each word/subword maps to a 768-dim vector. Why 768? Empirically found to balance expressiveness vs compute. Smaller (256) loses representation power; larger (1024+) has diminishing returns.
- **Segment Embedding**: BERT handles sentence pairs (for tasks like NLI). A/B segment IDs distinguish which sentence each token belongs to.
- **Position Embedding**: Transformers have no inherent notion of order (unlike RNNs). Learned position embeddings (not sinusoidal like original Transformer) let the model learn position-dependent patterns. Why learned? Simpler and works just as well up to 512 positions.

**Design trade-off**: Fixed 512 max positions. Longer sequences require sliding window or truncation — a limitation that decoder-only models later addressed with RoPE.

> **💡 FAQ: Why does position encoding need to be "learned"?**
>
> You might ask: isn't position information the same for all text? "Position 3" is just "position 3" — what's there to learn?
>
> Good intuition! Learned Position Embeddings don't learn "the position itself," but rather "the semantic tendency of that position" — e.g., position 1 often contains articles/subjects, positions 2-3 often contain adjectives, etc.
>
> But in practice, **both approaches perform similarly**. Even with Sinusoidal encoding, the Attention layers learn positional semantics anyway. BERT chose Learned mainly because **it's simpler to implement** (one line of `nn.Embedding` vs a math function), not because it's more effective.
>
> | | Sinusoidal | Learned |
> |--|-----------|---------|
> | Concept | Pure math formula | Learnable parameters |
> | Code | Requires a function | One line |
> | Length extrapolation | ✅ Yes | ❌ No |
> | Performance | About the same | About the same |

#### Component 2: Multi-Head Bidirectional Self-Attention

**What it does**: Each token computes attention over ALL other tokens simultaneously.

$$
\begin{aligned}
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\
\text{MultiHead}(X) &= \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O \\
\text{head}_i &= \text{Attention}(XW^Q_i, XW^K_i, XW^V_i)
\end{aligned}
$$

**Why bidirectional**: Consider the word "bank" in this sentence:
- "The river **bank** was steep and muddy"

To correctly encode "bank" as *riverbank* (not *financial institution*), the model needs:
- Left context: "The river" → suggests geographical meaning
- Right context: "was steep and muddy" → confirms physical terrain

Bidirectional attention lets BERT see both sides simultaneously, enabling this disambiguation. A left-only model (like GPT) would have to guess "bank" before seeing "was steep" — much harder.

**Why multi-head (12 heads)**: Different heads learn different attention patterns:
- Some heads attend to syntactic structure (subject-verb)
- Some attend to coreference (he → John)
- Some attend to adjacent tokens (local context)

12 heads × 64 dims = 768 total dims. More heads = more diverse patterns; but 16+ heads show diminishing returns.

**Design trade-off**: Bidirectional attention means O(n²) compute where n = sequence length. Why n²? Because each of the n tokens computes attention scores with ALL n tokens:

```
         t1    t2    t3    t4    t5
    ┌─────────────────────────────────┐
t1  │  ✓     ✓     ✓     ✓     ✓    │  ← t1 attends to 5 tokens
t2  │  ✓     ✓     ✓     ✓     ✓    │  ← t2 attends to 5 tokens
t3  │  ✓     ✓     ✓     ✓     ✓    │  ...
t4  │  ✓     ✓     ✓     ✓     ✓    │
t5  │  ✓     ✓     ✓     ✓     ✓    │
    └─────────────────────────────────┘
Total: n × n = n² attention scores
```

(Note: GPT's causal mask is still O(n²) — it just masks out the upper triangle, but the matrix is still n×n.)

And critically: **you cannot generate text efficiently**. Here's why:

In GPT (causal), adding a new token doesn't change the representations of previous tokens — they can't "see" the future anyway. So GPT caches the Key/Value matrices (KV-cache) and reuses them:

```
GPT generating "The cat sat on the" → "mat"
Step 1: Process "The cat sat on the", cache K,V ✓
Step 2: Generate "mat" — only compute Q for new token, reuse cached K,V
        → O(n) per token
```

In BERT (bidirectional), every token's representation depends on ALL other tokens, including ones added later. Adding a new token **changes all previous representations**:

```
BERT generating "The cat sat on the" → "?"
Step 1: "The cat sat on the [MASK]" — run full BERT, predict "mat"
Step 2: "The cat sat on the mat [MASK]" — must re-run ENTIRE BERT!
        Because adding "mat" changes:
        - "The"'s representation (it can now see "mat")
        - "cat"'s representation
        - ALL tokens change!
Step 3: Repeat for every new token...
        → O(n²) per token = O(n³) total for n tokens
```

**This is why all generative LLMs (ChatGPT, Claude, Gemini) use GPT-style architecture, not BERT.**

#### Component 3: Feed-Forward Network (FFN)

**What it does**: Applies non-linear transformation independently at each position.

```python
# FFN: expand to 4x, apply non-linearity, project back
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
# W₁: 768 → 3072 (4x expansion)
# W₂: 3072 → 768 (project back)
```

**Why 4× expansion**: The attention layer mixes information across positions, but it's linear. The FFN adds non-linearity and increases model capacity. Why 4×? Empirically optimal — larger gives diminishing returns, smaller loses expressiveness.

**Why GELU activation**: GELU (Gaussian Error Linear Unit) outperforms ReLU for language tasks. It's smooth (differentiable everywhere) and handles negative values better than ReLU.

![GELU vs ReLU](../zh/images/day05/gelu-vs-relu.png)
*Figure: GELU allows small negative gradients to flow, preventing "dead neurons" that plague ReLU.*

**Design trade-off**: FFN has the most parameters in each layer (2 × 768 × 3072 = 4.7M per layer). This is where most of the "knowledge" is stored.

#### Component 4: Residual Connection + LayerNorm

**What it does**: 
- Residual: `output = x + Sublayer(x)` — enables gradient flow through deep networks
- LayerNorm: Normalizes each sample independently for stable training

![Residual Connection](../zh/images/day05/residual-connection.png)
*Figure: Residual connections provide a "gradient highway" — even if Layer(x) gradients vanish, the shortcut path (×1) preserves gradient flow.*

**Why residual connections matter**: Without them, gradients must pass through every layer during backpropagation. If each layer shrinks the gradient by 0.9×, after 12 layers: 0.9¹² ≈ 0.28 — the gradient nearly vanishes! With residual connections, gradients have a shortcut that multiplies by 1, so: 1¹² = 1 — no vanishing.

**The math**:
```python
# Without residual: gradient shrinks
∂L/∂x = ∂L/∂y × ∂Layer/∂x        # ∂Layer/∂x might be small

# With residual: gradient has +1 term
y = x + Layer(x)
∂y/∂x = 1 + ∂Layer/∂x            # The "1" saves us!
```

**Why Post-LN (BERT) vs Pre-LN (GPT-2+)**: BERT uses Post-LN (normalize after residual). Later work found Pre-LN (normalize before sublayer) is more stable for very deep models. This became standard in GPT-2 onwards.

#### Training: Masked Language Modeling (MLM)

**What it does**: Randomly mask 15% of tokens, predict the originals.

**Why 15%**: 
- Too low (5%): Model sees too much context, task is too easy
- Too high (30%): Not enough context to make good predictions
- 15% was found optimal in ablation studies

**Why 80-10-10 masking strategy**:
- 80%: Replace with [MASK] token
- 10%: Replace with random token
- 10%: Keep original

This prevents the model from only learning to predict [MASK] tokens — it must handle corrupted and normal inputs.

```python
# BERT in action: masked token prediction
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The cat [MASK] on the mat."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0][1]
logits = outputs.logits[0, mask_idx]
predicted_token = tokenizer.decode([logits.argmax()])
print(f"Predicted: {predicted_token}")  # → "sat"
```

**BERT's Achilles heel**: Bidirectional attention makes generation inefficient. To generate token t+1, you'd need to mask it and re-encode the entire sequence. Generating n tokens requires n forward passes, each O(n²), for a total of **O(n³)** — compared to GPT's O(n²).

---

### 1.2 Decoder-only: GPT (2018–present)

GPT ([Radford et al., 2018](https://openai.com/research/language-unsupervised)) uses the Transformer decoder with one critical modification: **causal masking**.

> **💡 What is Causal Masking?**
> 
> Causal = "cause comes before effect." Each token can only attend to itself and **previous** tokens (the "past"), never future tokens. This is implemented as a lower-triangular attention matrix:
> ```
>            The   cat   sat   on
>       The   ✓     ✗     ✗     ✗    ← "The" only sees itself
>       cat   ✓     ✓     ✗     ✗    ← "cat" sees "The" + itself
>       sat   ✓     ✓     ✓     ✗    ← "sat" sees first three
>       on    ✓     ✓     ✓     ✓    ← "on" sees everything before it
> ```
> This mirrors real text generation: when writing a word, you only know what you've written so far.

#### Layer Structure (repeated L times)

```
Input Embeddings
       ↓
┌─────────────────────────────────────┐
│  Multi-Head CAUSAL Self-Attention   │  ← Only sees past tokens
│  + Residual Connection + LayerNorm  │
├─────────────────────────────────────┤
│  Feed-Forward Network (FFN)         │
│  + Residual Connection + LayerNorm  │
└─────────────────────────────────────┘
       ↓ (repeat L times)
Final Hidden States → Next-token prediction head
```

#### The Key Difference: Causal Attention Mask

**What it does**: A lower-triangular mask prevents position i from attending to positions j > i.

```python
# Causal mask for sequence length 5
# 1 = can attend, 0 = blocked
mask = [
    [1, 0, 0, 0, 0],  # Position 0: sees only itself
    [1, 1, 0, 0, 0],  # Position 1: sees 0-1
    [1, 1, 1, 0, 0],  # Position 2: sees 0-2
    [1, 1, 1, 1, 0],  # Position 3: sees 0-3
    [1, 1, 1, 1, 1],  # Position 4: sees 0-4 (all past)
]
```

**Why this design**: The "moving wall" mental model — at each position, you can only see what came before. This constraint:
1. Makes training and inference use the same computation
2. Enables efficient KV-cache at inference (more on this in Section 4)
3. Naturally models the autoregressive factorization of language

> **💡 What is "autoregressive factorization"?**
> 
> It's the chain rule decomposition of language probability:
> $$P(\text{The, cat, sat}) = P(\text{The}) \times P(\text{cat}|\text{The}) \times P(\text{sat}|\text{The, cat})$$
> 
> Each word's probability depends **only on previous words**. GPT's causal mask perfectly matches this:
> - Predict "The": see [START] → P(The)
> - Predict "cat": see "The" → P(cat|The)  
> - Predict "sat": see "The cat" → P(sat|The,cat)
>
> This is how humans write — word by word, only knowing what came before.

**Design trade-off**: No access to future context. The word "bank" in "I sat by the river bank" must be encoded without seeing "river" first. This seems like a disadvantage — but scale compensates.

#### Pre-LN vs Post-LN

**GPT-2 onwards uses Pre-LN**: LayerNorm BEFORE attention/FFN, not after.

```python
# Post-LN (BERT, GPT-1): LayerNorm AFTER the sublayer
x = LayerNorm(x + Attention(x))  # ❌ Unstable for deep models

# Pre-LN (GPT-2+): LayerNorm BEFORE the sublayer
x = x + Attention(LayerNorm(x))  # ✅ More stable gradients
```

**Why switch**: Pre-LN produces more stable gradients in very deep models (48+ layers). This became critical as models scaled to hundreds of billions of parameters.

#### Position Encoding: From Learned to RoPE

**GPT-1/2**: Learned position embeddings (like BERT), limited to 1024 positions.

**GPT-3+**: Still learned, but with extrapolation tricks.

> **💡 What are "extrapolation tricks"?**
> 
> Problem: GPT-3 was trained with max 2048 tokens. What if inference needs 4000 tokens? Position 2048+ was never seen during training!
> 
> Common tricks:
> - **Position Interpolation**: Compress positions (pos 4000 → treat as pos 2000)
> - **ALiBi**: Skip position embeddings; add attention bias based on distance
> - **NTK-aware Scaling**: Adjust the frequency base to "stretch" position encodings
> 
> These are patches, not native solutions — hence "tricks."

**Modern (LLaMA, etc.)**: RoPE (Rotary Position Embedding) — encodes relative positions through rotation matrices. Enables length generalization far beyond training length. Unlike learned embeddings, RoPE is designed for extrapolation from the start.

**Why the evolution**: Learned embeddings don't generalize beyond training length. RoPE's rotation-based approach naturally handles longer sequences.

#### Training: Causal Language Modeling (CLM)

**What it does**: Predict the next token at every position.

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, ..., x_{t-1})
$$

**Why this is efficient**: Every position provides a training signal. For a 1000-token sequence:
- BERT (15% masking): ~150 prediction tasks
- GPT (CLM): 1000 prediction tasks

**6–7× more efficient** in extracting signal from the same data.

> **💡 Why exactly 6-7×?**
> 
> ```
> Efficiency ratio = 1000 / 150 ≈ 6.67×
> ```
> 
> Visualized:
> ```
> Sentence: "The cat sat on the mat"
>            1   2   3   4   5   6
> 
> BERT (say position 3 is masked):
>   Prediction tasks: 1 (predict "sat")
> 
> GPT (CLM):
>   Position 1 → predict "cat"  ✓
>   Position 2 → predict "sat"  ✓
>   Position 3 → predict "on"   ✓
>   Position 4 → predict "the"  ✓
>   Position 5 → predict "mat"  ✓
>   Total: 5 prediction tasks
> ```
> 
> Same data, but GPT extracts **6-7× more learning signal**. To match GPT's learning, BERT needs ~7× more data or training time.

```python
# GPT in action: text generation
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompt = "The transformer architecture"
inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(
    inputs['input_ids'],
    max_new_tokens=30,
    do_sample=True,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

### 1.3 Encoder-Decoder: T5 (2020)

T5 ([Raffel et al., 2020](https://arxiv.org/abs/1910.10683)) keeps the full original Transformer architecture — separate encoder and decoder stacks connected by cross-attention.

#### Layer Structure

```
ENCODER (bidirectional, L layers):          DECODER (causal, L layers):
┌─────────────────────────────┐             ┌─────────────────────────────┐
│  Bidirectional Self-Attn    │             │  Causal Self-Attention      │
│  + Residual + LayerNorm     │             │  + Residual + LayerNorm     │
├─────────────────────────────┤             ├─────────────────────────────┤
│  Feed-Forward Network       │             │  Cross-Attention            │ ← Queries encoder
│  + Residual + LayerNorm     │             │  + Residual + LayerNorm     │
└─────────────────────────────┘             ├─────────────────────────────┤
        ↓ (L layers)                        │  Feed-Forward Network       │
                                            │  + Residual + LayerNorm     │
    Encoder Output ───────────────────────→ └─────────────────────────────┘
    (K, V for cross-attention)                      ↓ (L layers)
```

#### Cross-Attention: The Bridge

**What it does**: Decoder queries attend to encoder outputs.

```python
# In cross-attention:
Q = decoder_hidden @ W_Q  # Query from decoder
K = encoder_output @ W_K  # Key from encoder
V = encoder_output @ W_V  # Value from encoder

attention = softmax(Q @ K.T / sqrt(d_k)) @ V
```

**Why this design**: The encoder processes the full input bidirectionally (good for understanding), then the decoder generates output autoregressively while "looking at" the encoder's representation through cross-attention.

**Use case**: Ideal for seq2seq tasks — translation, summarization — where you need full input understanding before generating output.

#### Why T5 Didn't Win

Despite its elegance, T5's architecture has scaling disadvantages:

1. **Double the parameters for same capacity**: Encoder + Decoder means two separate stacks. A 12-layer T5 has similar parameters to a 24-layer GPT, but the GPT can be deeper (more layers = better).

2. **Inference complexity**: Must run full encoder pass BEFORE any decoding. Can't start generating until input is fully processed.

3. **Cross-attention overhead**: Each decoder layer has an extra attention operation (cross-attention) on top of self-attention.

4. **Can't easily do in-context learning**: The encoder/decoder split doesn't naturally support "prompt + completion" format that made GPT-3 so versatile.

#### T5's Text-to-Text Framework

T5's key insight was unifying all tasks as text-to-text:

```
# Classification
Input:  "mnli premise: ... hypothesis: ..."
Output: "entailment"

# Translation  
Input:  "translate English to German: Hello world"
Output: "Hallo Welt"

# Summarization
Input:  "summarize: [long article]"
Output: "[short summary]"
```

This was influential — GPT-3 adopted similar task framing. But GPT proved you don't need separate encoder/decoder to do this.

```python
# T5 in action
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Translation task
input_text = "translate English to German: Hello, how are you?"
inputs = tokenizer(input_text, return_tensors='pt')

outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# → "Hallo, wie geht es Ihnen?"
```

---

### Summary: Architecture Design Decisions

| Decision | BERT | GPT | T5 |
|----------|------|-----|-----|
| **Attention direction** | Bidirectional | Causal (left-only) | Encoder: bidir, Decoder: causal |
| **Position encoding** | Learned (512 max) | Learned → RoPE | Relative position bias |
| **LayerNorm placement** | Post-LN | Pre-LN (GPT-2+) | Pre-LN |
| **Training objective** | MLM (15% masked) | CLM (100% positions) | Span corruption |
| **Cross-attention** | No | No | Yes |
| **Generation efficient** | ❌ | ✅ (KV-cache) | ✅ but slower |
| **In-context learning** | ❌ | ✅ | Limited |

The key insight: **GPT's simplicity is a feature, not a bug**. One stack, one attention type, one training objective — and it scales better than the alternatives.


---

## 2. Attention Masks: The Core Difference

The single most important structural difference between BERT and GPT is the attention mask.

![Attention Masks](../zh/images/day05/attention-masks.png)
*Figure 2: Left — BERT's full attention matrix: every token attends to every other token (all ✓). Right — GPT's causal mask: lower-triangular, with future positions blocked (✗). This one difference determines everything.*

For a sequence of length *n*:

- **BERT**: attention matrix is fully dense — O(n²) operations, all positions attend to all positions
- **GPT**: attention matrix is lower-triangular — same O(n²) operations, but with future masked out

The mathematical consequence: BERT can't generate text efficiently, because generating token *t+1* requires the token to be present in the input (masked), which defeats the purpose of generation. GPT can generate efficiently because predicting position *t+1* only requires the already-computed representations of positions 1 through *t*.

---

## 3. Training Objectives: MLM vs CLM

### 3.1 The Math

Here are the formal objectives for both training paradigms:

$$
\begin{aligned}
\mathcal{L}_{\text{MLM}} &= -\sum_{i \in \mathcal{M}} \log P(x_i \mid x_{\backslash \mathcal{M}}) \quad &\text{(BERT: predict masked tokens)} \\
\mathcal{L}_{\text{CLM}} &= -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1}) \quad &\text{(GPT: predict next token)} \\
P(x_1, \ldots, x_T) &= \prod_{t=1}^{T} P(x_t \mid x_{<t}) \quad &\text{(chain rule factorization)}
\end{aligned}
$$

Where:
- $\mathcal{M}$ = set of masked token indices (for MLM)
- $x_{\backslash \mathcal{M}}$ = all non-masked tokens (visible context for BERT)
- $x_{<t}$ = all tokens before position *t* (left context for GPT)

**Key insight from the math**: MLM trains on only ~15% of tokens per sequence (the masked ones). CLM trains on *every single token position* — 100% training signal efficiency. This means GPT extracts 6–7× more training signal from the same amount of data.

![Training Objectives](../zh/images/day05/training-objectives.png)
*Figure 3: MLM vs CLM training objectives. BERT predicts masked tokens in parallel using full bidirectional context. GPT predicts each next token sequentially using only left context — but trains on every position.*

### 3.2 Why CLM Gets More Signal

Consider a sequence of 1000 tokens:
- BERT masks ~150 tokens → trains on 150 prediction tasks
- GPT trains on 1000 prediction tasks (every position predicts the next)

This isn't just about efficiency — it's about what the model learns. GPT must internalize the entire distributional structure of language to minimize CLM loss. It can't "cheat" by just copying visible context; it must actually model language.

---

## 4. Why GPT Won: Four Concrete Mechanisms

This is the crux of the article. The question isn't just "which is better" — it's *why* the mechanisms of decoder-only architectures proved decisive at scale.

![Why GPT Won](../zh/images/day05/why-gpt-won.png)
*Figure 4: Four concrete reasons why decoder-only architectures came to dominate. Each reason compounds the others — together they create a decisive advantage at scale.*

### 4.1 KV-Cache: The Inference Efficiency Killer Feature

When GPT generates token *t+1*, it needs attention over positions 1 through *t*. In standard attention, computing the Key (K) and Value (V) matrices for all previous positions at every new step would be O(n²) per token — brutally slow.

The solution: **KV-cache**. Since past positions never change (causal mask — position *t* only sees *t* and earlier), we cache the K and V matrices for all previous positions. Adding token *t+1* only requires computing K and V for that *one new position*, not all previous ones.

```python
# Pseudocode: KV-cache in action
class GPTWithKVCache:
    def __init__(self):
        self.kv_cache = []  # Cache of (K, V) for each layer
    
    def generate_one_token(self, new_token_embedding):
        # For each transformer layer:
        for layer in self.layers:
            # Compute K, V for NEW token only
            new_k = layer.W_k(new_token_embedding)
            new_v = layer.W_v(new_token_embedding)
            
            # Extend cache with new K, V
            self.kv_cache[layer].append((new_k, new_v))
            
            # Compute Q for new token, attend to ALL cached K, V
            new_q = layer.W_q(new_token_embedding)
            all_k = torch.cat([c[0] for c in self.kv_cache[layer]])
            all_v = torch.cat([c[1] for c in self.kv_cache[layer]])
            
            attn_output = attention(new_q, all_k, all_v)
            new_token_embedding = layer.ffn(attn_output)
        
        return new_token_embedding
```

**Why does BERT lack this?** BERT's bidirectional attention means every position can see every other position. If you change one token (e.g., append a new token), you potentially change all attention scores everywhere — the cache is invalidated. There is no simple KV-cache for bidirectional models.

This single advantage makes GPT-style generation dramatically faster at inference time. GPT's per-token generation cost is O(n) (attend to cached K,V), not O(n²) per step.

### 4.2 Training Parallelization: Full Sequence Utilization

During training, GPT processes the entire sequence in one forward pass due to the causal mask. All positions (1 through T) are trained simultaneously:

- Position 1 predicts position 2
- Position 2 predicts position 3
- ...
- Position T-1 predicts position T

This is massively GPU-parallel. Every single token in every single training document contributes gradients simultaneously. For a 1-trillion-token training corpus, GPT trains on 1 trillion prediction tasks. BERT trains on roughly 150 billion (15% × 1T).

The implication for scaling: GPT can be trained on larger datasets with greater computational efficiency. When you're pushing to 100B+ parameters, this efficiency gap becomes critical.

### 4.3 Unified Paradigm: One Architecture, All Tasks

BERT's bidirectionality is great for classification — but BERT needs a different head for every task:
- Text classification: [CLS] token → linear classifier
- Named entity recognition: each token → linear classifier
- Question answering: two pointers for start/end spans
- Machine translation: needs an entirely separate decoder architecture

GPT says: *every task is text generation*. The API for every task is identical:

```python
# Everything is text generation with GPT
tasks = {
    "translation":    "Translate to French: The cat is sleeping. →",
    "classification": "Sentiment of 'I loved this movie': positive or negative? →",
    "summarization":  "Summarize: [long article]... Summary:",
    "qa":             "Q: What is the capital of France? A:",
    "code":           "Write a Python function to reverse a string:\ndef reverse_string(",
}

for task, prompt in tasks.items():
    result = gpt.generate(prompt)
    print(f"{task}: {result}")
```

This unified paradigm means:
1. One model can handle unlimited task types without fine-tuning
2. New capabilities emerge from prompting alone
3. No architecture modifications for new tasks
4. Transfer across tasks happens naturally

### 4.4 In-Context Learning: The Emergent Scaling Reward

This is perhaps the most surprising mechanism. At sufficient scale (GPT-3's 175B parameters), decoder-only models develop the ability to **learn from examples provided directly in the input**:

```
# Zero-shot
Translate to French: The cat is sleeping.
Translation:

# Few-shot (in-context learning)
Translate to French: The dog runs. → Le chien court.
Translate to French: Birds fly. → Les oiseaux volent.
Translate to French: The cat is sleeping.
Translation:
```

The few-shot GPT-3 often matches or beats fine-tuned BERT-sized models on the same tasks — *without any gradient updates*. The model appears to perform task recognition and adaptation within a single forward pass.

Why does this emerge in decoder-only models and not encoder-only? The leading hypothesis: autoregressive training forces the model to solve an implicit meta-learning problem. To predict each next token, it must track what task is being described in the context, adapt its "strategy," and apply that strategy to new inputs. This meta-learning capacity is baked into CLM training at scale.

Bidirectional models like BERT don't develop this capability — there's no incentive during masked-token prediction to learn to read task specifications and apply them.

---

## 5. The Architecture Evolution Timeline

![Timeline](../zh/images/day05/timeline.png)
*Figure 5: Architecture evolution from 2017 to 2023. BERT initially dominated NLP benchmarks, but decoder-only models (GPT-2, GPT-3, ChatGPT) showed increasingly powerful capabilities at scale, ultimately winning the mainstream.*

The timeline tells the story clearly:

| Year | Model | Architecture | Significance |
|------|-------|-------------|--------------|
| 2017 | Transformer | Enc-Dec | Foundation: "Attention Is All You Need" |
| 2018 | BERT | Encoder-only | SOTA on 11 NLP benchmarks — BERT fever |
| 2018 | GPT-1 | Decoder-only | 117M params, promising but not yet viral |
| 2019 | GPT-2 | Decoder-only | 1.5B params — "too dangerous to release" |
| 2020 | T5 | Enc-Dec | Text-to-text unification, 11B params |
| 2020 | GPT-3 | Decoder-only | 175B params — in-context learning emerges |
| 2022 | ChatGPT | Decoder-only + RLHF | 100M users in 2 months |
| 2023 | LLaMA/Gemini/Claude | Decoder-only | Entire industry converges on decoder-only |

The tipping point was GPT-3 (2020). Its in-context learning capability, combined with the inference efficiency advantages of KV-cache, made scaling decoder-only models the obvious path forward. ChatGPT (2022) just made this mainstream-visible.

---

## 6. BERT Isn't Dead

A common misconception: "GPT won, so BERT is obsolete." This is wrong.

BERT's bidirectional encoding remains the best tool for several critical applications:

**Semantic Search & Retrieval**: BERT-style models (and their descendants like [Sentence-BERT](https://arxiv.org/abs/1908.10084)) generate rich contextual embeddings that capture meaning in a fixed-size vector. These power modern search systems:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Encode documents and queries for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Neural networks learn from data",
    "Transformers use attention mechanisms",
    "BERT is bidirectional",
]
query = "How does BERT process text?"

# Encode everything at once (batch processing)
doc_embeddings = model.encode(documents)
query_embedding = model.encode(query)

# Cosine similarity search
similarities = np.dot(doc_embeddings, query_embedding) / (
    np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
)
best_match = documents[np.argmax(similarities)]
print(f"Most relevant: {best_match}")
```

**Text Classification at Scale**: For production classification systems (spam detection, sentiment, intent classification), BERT-sized models fine-tuned on labeled data are often faster, cheaper, and more accurate than prompting a large GPT model.

**Retrieval-Augmented Generation (RAG)**: In modern RAG systems, the *retrieval* step often uses BERT-style encoders to find relevant documents, while GPT-style decoders generate the final answer. BERT and GPT work together.

**Why BERT survives**: Its bidirectional encoding creates better token/sentence embeddings for *understanding* tasks. GPT embeddings are less semantically tight because the model is optimized for generation, not similarity matching.

---

## 7. Common Misconceptions

### ❌ "BERT can be used for text generation too, just slowly"

Not really. BERT's fill-in-the-blank (MLM) can fill one masked token per forward pass, but this doesn't scale to coherent multi-sentence generation. The model was never trained to maintain narrative coherence across autoregressively generated text. True generation requires the causal structure of GPT.

### ❌ "T5 should have won because it combines the best of both"

T5 is powerful and still widely used (especially for specific seq2seq tasks). But the encoder-decoder split creates inference complexity: you need to run the full encoder before starting decoding, and there's no simple KV-cache for the cross-attention mechanism. At inference time, this overhead compounds. For conversational AI at scale, GPT's simpler architecture wins on deployment economics.

### ❌ "Decoder-only models don't understand text, they just predict tokens"

This conflates mechanism with capability. GPT-3 and later models demonstrate sophisticated language understanding (reading comprehension, logical reasoning, code analysis) despite being "just" next-token predictors. The CLM objective, applied at sufficient scale, forces the model to build deep world models to minimize prediction loss. Understanding emerges from generation training.

### ❌ "Bigger always beats architecture choice"

Architecture matters enormously. A 7B decoder-only model (e.g., LLaMA-2-7B) outperforms many earlier 13B models across benchmarks. The decoder-only architecture's training efficiency means each parameter is better utilized.

---

## 8. Code Example: Architecture in Practice

```python
# Side-by-side comparison: BERT vs GPT for the same task
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    GPT2LMHeadModel, GPT2Tokenizer,
    pipeline
)

# ---- BERT: Classification (its strength) ----
bert_classifier = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english'
)
text = "The transformer architecture changed everything about NLP"

# BERT processes the full text bidirectionally
result = bert_classifier(text)
print(f"BERT sentiment: {result[0]['label']} ({result[0]['score']:.3f})")

# ---- GPT: Generation (its strength) ----
gpt_generator = pipeline('text-generation', model='gpt2')

# GPT generates the next tokens autoregressively
generated = gpt_generator(
    "The transformer architecture changed everything about",
    max_new_tokens=20,
    do_sample=True,
    temperature=0.7,
    pad_token_id=50256  # GPT-2's EOS token
)
print(f"GPT completion: {generated[0]['generated_text']}")

# Key insight: these models excel at fundamentally different tasks
# Use BERT-family for: classification, NER, embeddings, retrieval
# Use GPT-family for: generation, conversation, in-context learning
```

---

## 9. Further Reading

### Beginner

1. [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) by Jay Alammar — Visual walkthrough of BERT's design
2. [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) by Jay Alammar — How GPT generates text, step by step

### Advanced

1. [Understanding Large Language Models](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/lectures/lec01.pdf) — Princeton COS 597G lecture notes
2. [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — Overview of attention efficiency improvements including KV-cache

### Papers

1. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., Google (2018)
2. [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., OpenAI (2019)
3. [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., OpenAI (2020)
4. [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683) — Raffel et al., Google (2020)
5. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Reimers & Gurevych (2019)

---

## Reflection Questions

1. **The KV-cache advantage**: Why does the causal mask enable KV-caching but bidirectional attention does not? What specifically about bidirectional attention invalidates a cached Key-Value matrix when a new token is appended?

2. **Training efficiency tradeoff**: CLM trains on 100% of token positions vs MLM's ~15%. Does this mean CLM always converges faster? Are there tasks where MLM's bidirectional training signal is actually *better* per token? Think about tasks where context is symmetric.

3. **The unified paradigm assumption**: GPT claims "all tasks = text generation." But some tasks are genuinely not text generation — for example, structured prediction (output a parse tree, a table, a protein sequence). How would you use a decoder-only model for these? What are the limits of the text-generation paradigm?

4. **BERT's future**: In a world where RAG systems use BERT for retrieval and GPT for generation, is there a single architecture that could do both well? What would it look like?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| **BERT** | Encoder-only; bidirectional attention; excellent for understanding, poor for generation |
| **GPT** | Decoder-only; causal mask; excellent for generation, scales to trillion-token training |
| **T5** | Encoder-Decoder; powerful seq2seq but complex inference |
| **MLM** | Train by predicting ~15% masked tokens — bidirectional but data-inefficient |
| **CLM** | Train by predicting next token at every position — 100% data efficiency |
| **KV-cache** | Cache K,V for past tokens; enabled by causal mask; makes GPT inference fast |
| **In-context learning** | Emerges at scale in decoder-only; adapt to new tasks from examples in prompt |
| **Why BERT survives** | Better embeddings for search/retrieval/classification; used in RAG retrieval step |

**Key Takeaway**: GPT's victory over BERT wasn't a matter of one architecture being "smarter" — it was four concrete mechanical advantages that compound at scale: KV-cache enables fast inference, CLM provides more training signal, the unified text-generation paradigm eliminates task-specific engineering, and in-context learning emerges naturally from autoregressive training. BERT isn't dead, but its domain is now clearly circumscribed: dense retrieval, classification, and embedding tasks where bidirectional understanding genuinely matters. Everything else has converged on decoder-only architectures.


---

## Appendix A: Why FFN Expands Then Compresses (768 → 3072 → 768)

> This appendix explores the mathematical foundations behind FFN's "expand-then-compress" design and connects it to modern model compression techniques.

### A.1 The Puzzle

At first glance, FFN's design seems contradictory:

```
768 dims → 3072 dims → 768 dims
    │          │          │
  input    expand 4×    back to original?
```

**Why expand to a higher dimension just to compress back?**

### A.2 The Mathematical Foundation

#### Cover's Theorem (1965)

> Points that are not linearly separable in low dimensions often become linearly separable when projected to higher dimensions through non-linear transformations.

**Example: XOR Problem**

In 2D, XOR is not linearly separable:

```
    1 │  ×       ○
      │
    0 │  ○       ×
      └───────────
         0       1

No single line can separate × from ○
```

Add a dimension z = x × y:

```
(0,0) → (0,0,0)  ×     Now a plane z=0.5
(1,1) → (1,1,1)  ×     can separate them!
(0,1) → (0,1,0)  ○
(1,0) → (1,0,0)  ○
```

**This is exactly what FFN does**: expand to 3072 dims where complex patterns become separable, apply non-linearity (GELU), then project the separated results back to 768 dims.

#### Johnson-Lindenstrauss Lemma (1984)

> High-dimensional points can be projected to lower dimensions while approximately preserving pairwise distances.

This explains why compressing 3072 → 768 doesn't lose critical information — the important structure is preserved.

#### Universal Approximation Theorem (1989)

> A single hidden layer neural network with sufficient width can approximate any continuous function.

FFN is literally a single-hidden-layer network:
- Input: 768
- Hidden: 3072 (the "sufficient width")
- Output: 768

### A.3 What Gets Lost in Compression?

**Yes, information is lost when projecting 3072 → 768.** But:

1. **W2 is learned, not random** — training teaches it to preserve task-relevant information
2. **Redundant dimensions are discarded** — the 3072-dim representation has redundancy
3. **Like JPEG compression** — discard what doesn't matter, keep what does

### A.4 Connection to Model Compression Paradigms

The "expand-compress" insight connects to three major model compression approaches:

| Paradigm | Core Idea | What It Compresses |
|----------|-----------|-------------------|
| **Pruning (Lottery Ticket)** | Large networks contain effective subnetworks | Network structure (remove neurons/connections) |
| **KAN** | Learn functions instead of weights | Representation (functions are more compact than weight matrices) |
| **Quantization (TurboQuant)** | Don't need full numerical precision | Bit precision (16-bit → 3-bit) |

#### Pruning: Finding the "Winning Ticket"

**Lottery Ticket Hypothesis (Frankle & Carlin, 2019)**:

> A randomly initialized large network contains a small subnetwork that, when trained in isolation with the same initialization, achieves comparable accuracy.

```
Large network (100% params) 
    → Train → 95% accuracy
    
Hidden inside: "winning ticket" (10% params)
    → Train with ORIGINAL init → 95% accuracy
    → Train with NEW random init → Much worse!
```

**Implication**: Most parameters in large models are "losing lottery tickets."

#### KAN: Learning Functions, Not Weights

**Kolmogorov-Arnold Networks (MIT, 2024)**:

Traditional MLP:
```
Fixed activation (GELU), learn weights
y = W2 · GELU(W1 · x)
```

KAN:
```
Fixed structure, learn activation functions
y = Σ φᵢ(xᵢ)   where φᵢ are learnable splines
```

| | MLP | KAN |
|--|-----|-----|
| Edges | Numbers (weights) | Functions (learnable) |
| Nodes | Apply activation | Just sum |
| Interpretability | Black box | Can visualize learned functions |
| Scientific discovery | Hard | Can recover formulas like E=½mv² |

#### Quantization: Fewer Bits, Same Information

**TurboQuant (Google, March 2026)**:

```
Original KV-cache: 16-bit floats
    ↓ TurboQuant
Compressed: 3-bit integers

Memory: ~6× reduction
Speed: ~8× faster on H100
Accuracy loss: Near zero
```

Uses **QJL (Quantized Johnson-Lindenstrauss)** — a 1-bit error checker based on the same JL Lemma that explains why dimension reduction preserves information.

### A.5 Can These Be Combined?

Theoretically yes:

```
Original model (100% params, 16-bit, MLP)
    ↓ Pruning
Subnetwork (10% params, 16-bit, MLP)
    ↓ Quantization  
Compressed (10% params, 3-bit, MLP)
    ↓ KAN (future?)
Ultimate (10% params, 3-bit, KAN)
```

This could yield 60×+ compression. Research ongoing.

### A.6 The Philosophical Divide

Each approach embeds a different belief about neural networks:

| Approach | Belief |
|----------|--------|
| **Pruning** | "Most neurons are redundant" |
| **KAN** | "Functions are more fundamental than numbers" |
| **Quantization** | "We don't need that much numerical precision" |
| **FFN design** | "Complex patterns need high-dim space to untangle" |

### A.7 Key Takeaways

1. **FFN's expand-compress is mathematically grounded** — Cover's theorem explains why high dimensions help; JL lemma explains why compression works

2. **"Losing information" is okay if you lose the right information** — learned projections preserve what matters

3. **Three compression paradigms** target different redundancies:
   - Structural (pruning)
   - Representational (KAN)  
   - Numerical (quantization)

4. **The future likely combines all three** — achieving extreme compression while maintaining capability

*This appendix emerged from a discussion about why FFN seems to "waste" computation by expanding then compressing. The answer: it's not waste, it's the mathematical foundation of how neural networks separate complex patterns.*

---

## Appendix B: LayerNorm Explained Visually

**LayerNorm** normalizes each layer's output to mean=0, std=1, keeping values stable across deep networks.

### The Formula

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
$$

Where:
- μ = mean of x
- σ = standard deviation of x
- γ, β = learnable parameters (scale and shift)

### Step-by-Step Example

![LayerNorm Explained](../zh/images/day05/layernorm-explained.png)

```python
x = [2.0, 4.0, 6.0, 8.0]  # Raw input

# Step 1: Calculate mean
μ = (2 + 4 + 6 + 8) / 4 = 5.0

# Step 2: Calculate std
σ = √[((2-5)² + (4-5)² + (6-5)² + (8-5)²) / 4] = 2.24

# Step 3: Normalize
x_norm = [(2-5)/2.24, (4-5)/2.24, (6-5)/2.24, (8-5)/2.24]
       = [-1.34, -0.45, +0.45, +1.34]

# Result: mean ≈ 0, std ≈ 1 ✓
```

### Why Not Just Divide by Sum?

| Method | Formula | Result | Problem |
|--------|---------|--------|---------|
| Divide by sum | x / Σx | [0.1, 0.2, 0.3, 0.4] | Only ensures sum=1, unstable |
| **LayerNorm** | (x-μ)/σ | [-1.34, -0.45, 0.45, 1.34] | mean=0, std=1, stable! |

**The problem with sum=1**: It only controls the *total*, not the *distribution shape*:

```python
# Case 1: Relatively uniform values
x = [100, 100, 100, 100]
x / sum(x) = [0.25, 0.25, 0.25, 0.25]  ✓ OK

# Case 2: One extreme value
x = [1000, 1, 1, 1]
x / sum(x) = [0.997, 0.001, 0.001, 0.001]  ← One dominates!

# Case 3: Very small values
x = [0.001, 0.002, 0.001, 0.001]
x / sum(x) = [0.2, 0.4, 0.2, 0.2]  ← Magnified 200×!
```

**LayerNorm always produces the same distribution shape**:

```python
# Case 2 with LayerNorm: extreme value gets "pulled back"
x = [1000, 1, 1, 1]
LayerNorm(x) = [1.73, -0.58, -0.58, -0.58]  ← Within ±2 range!

# Case 3 with LayerNorm: small values get same treatment
x = [0.001, 0.002, 0.001, 0.001]
LayerNorm(x) = [-0.58, 1.73, -0.58, -0.58]  ← Same range!
```

**Sum=1 controls "total amount"; LayerNorm pins the distribution to N(0,1).**

### Why γ and β?

After normalization, all layers output mean=0, std=1. But sometimes the model **needs** a different distribution:

```python
y = γ * x_norm + β  # Final output

# γ=1, β=0: Keep normalized result
# γ=2, β=3: Let model learn optimal scale/shift
```

**γ and β are learnable** — the model decides what distribution each layer needs.

#### Example: Sentiment Intensity

Suppose a layer learns "sentiment intensity":

```python
# Raw layer output (before LayerNorm)
"I love this!"   → [+5.0]   # Strong positive
"It's okay"      → [+0.5]   # Neutral
"I hate this!"   → [-5.0]   # Strong negative
```

**Without γ, β**: forced to N(0,1), differences get compressed:
```python
# After LayerNorm only
"I love this!"   → [+1.0]   # Was 10× different from hate,
"I hate this!"   → [-1.0]   # now only 2× different!
```

**With γ=5, β=0**: model recovers the needed range:
```python
# After LayerNorm × γ + β
"I love this!"   → [+5.0]   # Original intensity restored!
"I hate this!"   → [-5.0]   
```

#### Example: After ReLU

ReLU outputs are always ≥ 0. What if the next layer needs negative values?

```python
# ReLU output: [0, 1, 3] (all non-negative)
# After LayerNorm: [-1, 0, 1]
# With γ=1, β=-2: [-3, -2, -1]  ← Now we have negatives!
```

#### The Pattern: "Compress Then Expand"

| Step | Purpose |
|------|---------|
| LayerNorm | "Compress" to N(0,1) for numerical stability |
| γ (scale) | Let model decide how much "variance" this layer needs |
| β (shift) | Let model decide where the "center" should be |

**Why not skip LayerNorm entirely?**
- No LayerNorm → values may explode → training crashes
- LayerNorm without γ,β → stable but limited expressiveness
- LayerNorm + γ,β → stable AND expressive ✓

### One-liner

**LayerNorm = (x - mean) / std** — "flattens" uneven values to a standard range, keeping training stable across 12, 24, or 96 layers.

---

*Day 5 of 60 | LLM Fundamentals*
*Word count: ~5500 | Reading time: ~28 minutes*
