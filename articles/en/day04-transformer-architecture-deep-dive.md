# Day 4: Transformer Architecture Deep Dive

> **Core Question**: How do self-attention, multi-head attention, and positional encoding work together to build the Transformer?

---

## Opening

On June 12, 2017, a paper appeared on arXiv with an audacious title: "Attention Is All You Need." The claim seemed almost naive—surely you need *something* besides attention? Convolutions? Recurrence? *Something?*

But the authors weren't bluffing. They had built an architecture using only attention mechanisms, and it crushed the state-of-the-art in machine translation. The model was called the **Transformer**.

![Transformer Architecture](../zh/images/day04/transformer-architecture-v2.png)
*Figure 1: The complete Transformer architecture. Encoder (left) processes the input sequence; Decoder (right) generates the output sequence. Cross-attention connects them, allowing the decoder to "look at" encoder outputs. Residual connections (orange lines) help gradient flow.*

What made this architecture so revolutionary? And why has every major language model since—BERT, GPT, T5, LLaMA, Claude—been built on Transformers?

Yesterday we learned about attention: the mechanism that lets a model dynamically focus on relevant parts of the input. Today we go deeper. We'll understand:

1. **Self-attention vs. Cross-attention**: What's the difference, and when do you use each?
2. **Multi-head attention**: Why use multiple attention "heads," and what do they learn?
3. **Positional encoding**: How does a Transformer know word order without recurrence?
4. **The full architecture**: How these pieces combine into the encoder-decoder structure

By the end, you'll understand not just *what* a Transformer does, but *why* each component exists.

---

## 1. Self-Attention: Tokens Talking to Each Other

### 1.1 The Key Insight

In Day 3, we saw attention in the context of translation: a decoder attends to encoder outputs to decide which source words matter for the current target word. That's **cross-attention**—queries from one sequence, keys/values from another.

But the Transformer introduced something more powerful: **self-attention**, where a sequence attends to itself.

Consider the sentence: "The cat sat on the mat because **it** was tired."

What does "it" refer to? A human knows it's "the cat," not "the mat." But how does a model figure this out?

With self-attention, when processing "it," the model computes attention weights over all previous words. If trained properly, it learns to assign high weight to "cat"—because semantically, cats get tired, mats don't.

![Self-attention vs Cross-attention](../zh/images/day04/self-vs-cross-attention.png)
*Figure 2: Self-attention (left): each word attends to ALL words in the same sequence, including itself. Cross-attention (right): decoder words attend to encoder words, enabling translation alignment.*

### 1.2 The Mechanics

Self-attention is the same attention mechanism from Day 3, but Q, K, and V all come from the **same sequence**:

```
Input: X = [x₁, x₂, ..., xₙ]  (each xᵢ is an embedding vector)

Q = XW_Q    # Queries: "what am I looking for?"
K = XW_K    # Keys: "what do I contain?"  
V = XW_V    # Values: "what do I provide?"

Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The result is a new representation for each position, enriched with information from all other positions.

> **Library Analogy Revisited**
> 
> In self-attention, you're not searching a library—you're in a group discussion:
> - **Query**: Your question ("What does 'it' refer to?")
> - **Keys**: Everyone's expertise tags ("I know about cats", "I know about mats")
> - **Values**: What each person actually contributes when asked
> 
> You ask your question, check everyone's tags, and weight their contributions by relevance.

### 1.3 Why Self-Attention Matters

Self-attention gives the Transformer two superpowers that RNNs lacked:

**1. Global context in one step**

RNNs process sequentially—information from word 1 must pass through words 2, 3, ... to reach word 100. By then, it's diluted.

Self-attention connects every word to every other word directly. Word 100 can "see" word 1 without any intermediate steps. The path length is O(1), not O(n).

**2. Parallelization**

RNNs are inherently sequential—you can't compute h₁₀₀ until you've computed h₁ through h₉₉.

Self-attention computes all positions simultaneously. On a GPU, this is massively faster.

| Property | RNN | Self-Attention |
|----------|-----|----------------|
| Path length (word 1 → word 100) | O(n) | O(1) |
| Parallelizable | No | Yes |
| Training speed | Slow | Fast |
| Long-range dependencies | Struggles | Handles well |

### 1.4 Context Window: The Price of Self-Attention

The "context window" you hear about (4K, 8K, 128K tokens) is fundamentally **limited by self-attention**.

**Why?** Self-attention has O(n²) complexity:

```
n tokens, each attends to all n tokens
→ n × n = n² attention scores
→ n × n attention matrix stored in memory
```

| Context Length | Attention Matrix Size | Memory (FP16) |
|----------------|----------------------|---------------|
| 2K | 2K × 2K = 4M | ~8 MB |
| 8K | 8K × 8K = 64M | ~128 MB |
| 128K | 128K × 128K = 16B | ~32 GB |
| 1M | 1M × 1M = 1T | ~2 TB 😱 |

> **Context window = the maximum range self-attention can "see"**, limited by O(n²) memory and compute.

**How do we extend it?**

| Method | Idea |
|--------|------|
| **Sparse Attention** | Don't compute full n²—use local + global patterns |
| **Flash Attention** | Optimize I/O, avoid storing full matrix |
| **Ring Attention** | Distribute sequence across devices |
| **RoPE extrapolation** | Extend positional encoding beyond training length |
| **Linear Attention** | Replace O(n²) with O(n) |

We'll explore these in later chapters on efficient Transformers.

---

## 2. Multi-Head Attention: Attending in Different Ways

### 2.1 Why Multiple Heads?

Single-head attention computes one set of attention weights. But language has multiple types of relationships:

- **Syntactic**: "cat" relates to "sat" (subject-verb)
- **Semantic**: "cat" relates to "it" (coreference)
- **Positional**: "sat" relates to "on" (preposition follows verb)
- **Negation**: "not" modifies "happy"

One attention pattern can't capture all of these. What if we ran attention multiple times, in parallel, with different learned projections?

That's multi-head attention.

### 2.2 How It Works

Instead of one large attention, we use `h` smaller attentions (heads), each with dimension `d_k = d_model / h`:

![Multi-head Attention](../zh/images/day04/multi-head-attention.png)
*Figure 3: Multi-head attention with 8 heads. Input (d=512) is projected into 8 parallel attention computations (d_k=64 each). Outputs are concatenated and projected back to d=512. Each head can specialize in different relationship types.*

```python
# Typical configuration: d_model=512, h=8, d_k=64
d_model = 512
h = 8
d_k = d_model // h  # = 64

# Each head has its own W_Q, W_K, W_V of shape (d_model, d_k)
# Head i computes: head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
# Final output: Concat(head_1, ..., head_h) @ W_O
```

The math:

$$
\begin{aligned}
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
\end{aligned}
$$

> **What does Concat mean?**
> 
> Concat = Concatenate = join vectors end-to-end:
> ```
> Head 1 output: [a, b, c, d]       (64 dims)
> Head 2 output: [e, f, g, h]       (64 dims)
> ...
> Head 8 output: [w, x, y, z]       (64 dims)
> 
> After Concat: [a,b,c,d, e,f,g,h, ..., w,x,y,z]  (512 dims)
> ```
> 
> Multi-head flow: **split → parallel attention → concat back → project**

### 2.3 What Do Different Heads Learn?

Research has analyzed what different attention heads learn:

| Head Type | What It Attends To | Example |
|-----------|-------------------|---------|
| Syntactic | Subject-verb-object relations | "cat" → "sat" |
| Positional | Adjacent or nearby tokens | Position i → position i±1 |
| Separator | Punctuation, sentence boundaries | Words → [SEP] token |
| Coreference | Pronouns to their referents | "it" → "cat" |
| Rare token | Uncommon words (high information) | Any → "unprecedented" |

Different heads specialize automatically through training—nobody hardcodes these patterns.

### 2.4 Trade-off: Depth vs. Heads

Why not use 512 heads with d_k=1? Or 1 head with d_k=512?

**Too many heads** (small d_k): Each head has too few dimensions to learn meaningful patterns. A 1-dimensional attention can only represent trivial relationships.

**Too few heads** (large d_k): Not enough diversity in attention patterns. You lose the benefit of parallel perspectives.

**The sweet spot** in practice:
- GPT-2: 12 heads, d_k=64
- BERT: 12 heads, d_k=64
- GPT-3: 96 heads, d_k=128

The ratio d_model/h is typically between 64 and 128.

### 2.5 Multi-Head vs Context Window: A Common Confusion

**Q: If a model has 1M context window and 8 heads, does each head handle 1M/8 = 125K tokens?**

**A: No!** Each head sees the **full** 1M tokens. Multi-head splits **feature dimensions**, not **sequence length**.

```
Input: 1M tokens × 512 dimensions

                    1M tokens
                        ↓
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
 [Head 1]           [Head 2]     ...    [Head 8]
 1M × 64            1M × 64             1M × 64
 (full seq,         (full seq,          (full seq,
  fewer dims)        fewer dims)         fewer dims)
    ↓                   ↓                   ↓
    └───────────Concat──┼───────────────────┘
                        ↓
                 1M × 512
```

> **Key distinction:**
> - **n (sequence length)** = how many tokens (1M) → determines context window
> - **d_k (feature dimension)** = how many numbers describe each token (64) → split across heads
> 
> The attention matrix is **n × n** (1M × 1M), which is why context window is limited by O(n²).
> The d_k only affects the "richness" of each head's representation, not the context length.

**One sentence:** Multi-head doesn't divide the context window—it divides the feature space. Every head sees all tokens.

---

## 3. Positional Encoding: Teaching Order to a Bag

### 3.1 The Problem

Self-attention has a fatal flaw: **it doesn't know position**.

Consider: "dog bites man" vs. "man bites dog"

To self-attention without positional information, both are the same—a bag of {dog, bites, man}. The attention mechanism is **permutation invariant**: shuffling input order doesn't change output.

But word order matters! The two sentences have opposite meanings.

![Why Position Matters](../zh/images/day04/positional-encoding.png)
*Figure 4: Top-left: Same words, different order, different meaning. Top-right: Sinusoidal encoding heatmap—different positions have unique patterns. Bottom-left: Different dimensions capture different "frequencies" of position. Bottom-right: Word embedding + position encoding = position-aware input.*

### 3.2 The Solution: Add Position Information

The Transformer's solution is elegant: **add** a position encoding to each word embedding.

```
input_to_transformer = word_embedding + positional_encoding
```

Each position p gets a unique d_model-dimensional vector. When added to the word embedding, the result uniquely identifies both *what* the token is and *where* it is.

### 3.3 Sinusoidal Positional Encoding

The original Transformer used sinusoidal functions:

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\end{aligned}
$$

Where:
- `pos` = position in sequence (0, 1, 2, ...)
- `i` = dimension index (0, 1, 2, ..., d_model/2 - 1)
- Even dimensions use sine, odd dimensions use cosine

**Why sines and cosines?**

1. **Bounded values**: Always between -1 and 1, same scale as embeddings
2. **Unique patterns**: Each position has a distinct combination of sine values
3. **Relative positions**: PE(pos+k) can be expressed as a linear function of PE(pos) for any fixed k—this helps the model learn relative relationships
4. **Generalization**: Can extrapolate to longer sequences than seen in training (theoretically)

**The frequency intuition:**

Different dimensions oscillate at different frequencies:
- Dimension 0: Very slow oscillation (changes gradually across positions)
- Dimension 256: Fast oscillation (changes rapidly)

It's like representing position in a mixed radix system. Think of how we represent time: hours change slowly, minutes change faster, seconds change fastest. Position encoding works similarly—different dimensions capture different "scales" of position.

### 3.4 Learned vs. Fixed Positional Encoding

The original Transformer used fixed sinusoidal encodings. But you could also **learn** the position embeddings as parameters:

```python
# Learned positional encoding
position_embedding = nn.Embedding(max_seq_len, d_model)
# Each position 0, 1, 2, ... has a learned d_model-dimensional vector
```

**Comparison:**

| Approach | Pros | Cons |
|----------|------|------|
| Sinusoidal (fixed) | Extrapolates to longer sequences; fewer parameters | May not capture task-specific position patterns |
| Learned | Can learn task-specific patterns | Can't extrapolate beyond max_seq_len; more parameters |

In practice:
- **GPT-2, BERT**: Learned positional embeddings
- **Original Transformer**: Sinusoidal
- **Modern LLMs (GPT-4, LLaMA)**: RoPE (Rotary Position Embedding)—a clever hybrid we'll cover later

---

## 4. The Full Architecture

### 4.1 Encoder: Understanding the Input

The encoder's job is to build a rich representation of the input sequence.

**Structure (repeated N times):**

```
Input Embedding + Positional Encoding
        ↓
┌─────────────────────────┐
│   Multi-Head            │ ← Self-attention over input
│   Self-Attention        │
├─────────────────────────┤
│   Add & Norm            │ ← Residual + LayerNorm
├─────────────────────────┤
│   Feed-Forward          │ ← Position-wise transformation
│   Network               │
├─────────────────────────┤
│   Add & Norm            │ ← Residual + LayerNorm
└─────────────────────────┘
        ↓
    (repeat N times)
        ↓
    Encoder Output
```

**Key points:**

1. **Residual connections**: Add the input to the output of each sub-layer. This helps gradients flow and enables training very deep networks.

2. **Layer normalization**: Normalize activations to stabilize training. Applied after each residual connection.

3. **Feed-forward network**: A two-layer MLP applied independently to each position:

![Feed-Forward Network](../zh/images/day04/feed-forward-network.png)
*Figure 5: The position-wise feed-forward network. Input is expanded from d=512 to d=2048, passed through GELU activation, then projected back to d=512. Each position is processed independently—no cross-position information here.*

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

The FFN expands to 4x the model dimension (512 → 2048), applies nonlinearity, then projects back. This expansion allows the model to learn complex position-wise transformations.

### 4.2 Decoder: Generating the Output

The decoder generates output tokens one at a time, attending to both itself (past outputs) and the encoder (input).

**Structure (repeated N times):**

```
Output Embedding + Positional Encoding
        ↓
┌─────────────────────────┐
│   Masked Multi-Head     │ ← Self-attention (can't see future!)
│   Self-Attention        │
├─────────────────────────┤
│   Add & Norm            │
├─────────────────────────┤
│   Multi-Head            │ ← Cross-attention to encoder
│   Cross-Attention       │
│   (Q from decoder,      │
│    K,V from encoder)    │
├─────────────────────────┤
│   Add & Norm            │
├─────────────────────────┤
│   Feed-Forward          │
│   Network               │
├─────────────────────────┤
│   Add & Norm            │
└─────────────────────────┘
        ↓
    (repeat N times)
        ↓
    Linear + Softmax
        ↓
    P(next token)
```

### 4.3 The Causal Mask: No Peeking!

During training, we feed the decoder the entire target sequence. But it shouldn't be able to "peek" at future tokens—that would be cheating!

The **causal mask** (also called "look-ahead mask") blocks attention to future positions:

![Attention Masks](../zh/images/day04/attention-masks.png)
*Figure 6: Bidirectional attention (left) allows each position to see all others—used in encoders. Causal attention (right) only allows seeing past and present—used in decoders. The mask sets future positions to -∞ before softmax, making their attention weights 0.*

Mathematically, we add -∞ to attention scores for positions j > i before softmax:

```python
# Create causal mask
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
# [0, -inf, -inf, -inf]
# [0,  0,   -inf, -inf]
# [0,  0,    0,   -inf]
# [0,  0,    0,    0  ]

# Apply mask to attention scores
scores = (Q @ K.T) / sqrt(d_k)
scores = scores + mask  # Future positions become -inf
weights = softmax(scores)  # -inf → 0 weight
```

After softmax, positions with -∞ scores get 0 attention weight—they're effectively invisible.

### 4.4 Cross-Attention: The Bridge

Cross-attention connects encoder and decoder. The decoder's representation becomes the Query, while the encoder's output provides Keys and Values:

```
Q = decoder_hidden @ W_Q  # "What do I need?"
K = encoder_output @ W_K  # "What's available?"
V = encoder_output @ W_V  # "What to retrieve?"
```

This is exactly the attention mechanism from Day 3's translation example—the decoder "asks" the encoder which input words are relevant for generating the current output word.

---

## 5. Why This Architecture Works

### 5.1 Division of Labor

| Component | Responsibility |
|-----------|---------------|
| Self-attention | Model relationships between tokens |
| Multi-head | Capture different types of relationships in parallel |
| Position encoding | Inject sequence order information |
| FFN | Add nonlinearity and position-wise processing |
| Residual + LayerNorm | Enable deep stacking, stable training |
| Cross-attention | Bridge encoder and decoder |

### 5.2 Computation and Memory

The dominant cost is self-attention, which scales as O(n²) where n is sequence length:

- Every token attends to every other token: n × n attention matrix
- For sequence length 1000: 1 million attention scores per head
- For sequence length 100,000: 10 billion attention scores per head

This quadratic scaling is the Transformer's Achilles' heel. In Week 6, we'll learn about FlashAttention and sparse attention methods that mitigate this.

### 5.3 Why Encoder-Decoder for Translation?

The encoder-decoder structure maps naturally to sequence-to-sequence tasks:

1. **Encoder**: Process entire input, build understanding
2. **Cross-attention**: Align input to output
3. **Decoder**: Generate output, conditioned on input understanding

For tasks like text generation (GPT), only the decoder is needed—there's no separate input sequence.

For tasks like classification (BERT), only the encoder is needed—there's no generated output sequence.

This modularity is why Transformers became the backbone of nearly every NLP architecture.

---

## 6. Code Example

A simplified multi-head self-attention implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. Linear projections: (batch, seq_len, d_model)
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)
        
        # 2. Reshape to (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled dot-product attention
        # scores: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over last dimension (keys)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4. Apply attention to values
        # output: (batch, n_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, V)
        
        # 5. Concatenate heads: (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 6. Final linear projection
        output = self.W_O(output)
        
        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


# Demo
if __name__ == "__main__":
    # Create a simple test
    batch_size, seq_len, d_model = 2, 10, 512
    
    # Random input embeddings
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Positional encoding (simplified - just learned embeddings)
    position_enc = nn.Embedding(seq_len, d_model)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    x = x + position_enc(positions)
    
    # Encoder layer
    encoder = TransformerEncoderLayer(d_model=512, n_heads=8)
    output = encoder(x)
    
    print(f"Input shape:  {x.shape}")       # [2, 10, 512]
    print(f"Output shape: {output.shape}")  # [2, 10, 512]
    print("✓ Self-attention preserves sequence dimensions")
```

Key implementation notes:

1. **View and transpose** for multi-head: We reshape `(batch, seq, d_model)` to `(batch, n_heads, seq, d_k)` to process all heads in parallel
2. **Masked fill**: Use `-inf` to zero out forbidden positions after softmax
3. **Concatenate heads**: Reshape back to `(batch, seq, d_model)` before final projection
4. **Residual connections**: Add input to output of each sublayer

---

## 7. Math Derivation [Optional]

> This section is for readers who want deeper understanding. Feel free to skip.

### 7.1 Attention as Soft Dictionary Lookup

Think of attention as a differentiable dictionary lookup:

$$
\begin{aligned}
\text{Hard lookup: } &\text{output} = \text{dict}[\text{query}] \quad &\text{(exact match)} \\
\text{Soft lookup: } &\text{output} = \sum_i \alpha_i \cdot \text{value}_i \quad &\text{(weighted combination)}
\end{aligned}
$$

Where α_i = similarity(query, key_i) after softmax normalization.

### 7.2 Why Scaled Dot-Product?

The attention function is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**Why divide by √d_k?**

Without scaling, dot products grow with dimension:

$$
\begin{aligned}
q \cdot k &= \sum_{i=1}^{d_k} q_i k_i \\
\text{Var}(q \cdot k) &= d_k \cdot \text{Var}(q_i) \cdot \text{Var}(k_i) = d_k \quad &\text{(assuming unit variance)}
\end{aligned}
$$

For d_k = 64, dot products have variance 64, meaning standard deviation ≈ 8. Some scores might be 20+, others might be -20+.

After softmax, exp(20) ≈ 485 million while exp(-20) ≈ 0. The result is nearly one-hot: almost all weight on one key.

One-hot attention means:
- No gradient flows to other keys (gradients of 0-weight positions are 0)
- The model can't learn to consider multiple positions
- Effectively defeats the purpose of soft attention

Dividing by √d_k rescales variance back to 1, keeping softmax in a reasonable range.

### 7.3 Multi-Head as Ensemble

Multi-head attention can be viewed as an ensemble of attention functions:

$$
\begin{aligned}
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \in \mathbb{R}^{n \times d_k} \\
\text{MultiHead} &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \in \mathbb{R}^{n \times d_{model}}
\end{aligned}
$$

Each head operates in a different d_k-dimensional subspace. W^O combines these subspaces back to d_model.

Parameter count comparison (assuming d_model = h × d_k):
- Single head: 3 × d_model² + d_model² = 4d_model² parameters
- Multi-head: 3 × h × d_model × d_k + d_model² = 3d_model² + d_model² = 4d_model² parameters

Same parameter count, but multi-head provides multiple attention patterns!

### 7.4 Why Positional Encoding is Additive

Position information could be added or concatenated:

**Concatenation**: [word_embedding; position_encoding] → doubles dimension

**Addition**: word_embedding + position_encoding → same dimension

The Transformer uses addition because:
1. Maintains original model dimension (efficient)
2. Works well in practice
3. Position and content information can interact through subsequent layers

Interestingly, learned embeddings can partially separate these: the model can learn to "reserve" some dimensions for position and others for content.

---

## 8. Common Misconceptions

### ❌ "Transformer is just attention"

No! The Transformer is an *architecture* with many components:

| Component | Purpose |
|-----------|---------|
| Multi-head attention | Model relationships |
| Feed-forward network | Position-wise transformation |
| Residual connections | Enable deep networks |
| Layer normalization | Stabilize training |
| Positional encoding | Inject position information |

Attention alone wouldn't work—you need FFN for nonlinearity, residuals for depth, normalization for stability.

### ❌ "More heads = better"

Not necessarily. Research shows:

- **Some heads are redundant**: Many can be pruned with minimal accuracy loss
- **Head diversity matters**: Heads learning the same pattern are wasteful
- **Optimal head count depends on task**: More isn't always better

GPT-2's 12 heads work well; adding more doesn't proportionally improve performance.

### ❌ "Position encoding limits sequence length"

For **learned** positional embeddings: Yes, you can't extrapolate beyond max_seq_len.

For **sinusoidal** encodings: In theory, they generalize to any length. In practice, the model still struggles with lengths far beyond training distribution.

Modern solutions like **RoPE** (Rotary Position Embedding) and **ALiBi** (Attention with Linear Biases) better handle length extrapolation—we'll cover these in later articles.

### ❌ "Self-attention sees all positions equally"

Self-attention *can* see all positions, but doesn't weight them equally. The attention weights learn which positions matter:

- A pronoun might heavily attend to its referent
- A verb might attend to its subject and object
- Some positions get near-zero attention

The model learns to be selective, not uniform.

---

## 9. Further Reading

### Beginner
1. **The Illustrated Transformer** (Jay Alammar)
   Best visual explanation of Transformer architecture
   https://jalammar.github.io/illustrated-transformer/

2. **Attention? Attention!** (Lilian Weng)
   Comprehensive overview of attention mechanisms
   https://lilianweng.github.io/posts/2018-06-24-attention/

### Advanced
3. **The Annotated Transformer** (Harvard NLP)
   PyTorch implementation with line-by-line explanation
   http://nlp.seas.harvard.edu/annotated-transformer/

4. **A Mathematical Introduction to Transformers** (Phuong & Hutter)
   Rigorous mathematical treatment
   https://arxiv.org/abs/2312.10794

### Papers
5. **Attention Is All You Need** (Vaswani et al., 2017)
   The original Transformer paper
   https://arxiv.org/abs/1706.03762

6. **What Do Attention Heads Learn?** (Clark et al., 2019)
   Analysis of what different heads learn in BERT
   https://arxiv.org/abs/1906.04341

---

## Reflection Questions

1. **Self-attention is O(n²) in sequence length. For a 100K token context (like modern LLMs), how many attention scores must be computed per head? What strategies might help?**

2. **If position encoding is added to word embeddings, information might "interfere." How might the model learn to keep them separate? What happens if you use concatenation instead?**

3. **The decoder uses both self-attention and cross-attention. Could you build a translation model with only self-attention? How?** (Hint: Think about how modern decoder-only models handle translation.)

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| Self-attention | Sequence attends to itself, modeling internal relationships |
| Cross-attention | One sequence (decoder) attends to another (encoder) |
| Multi-head attention | Multiple parallel attention patterns, each learning different relationships |
| Positional encoding | Injects position information into a position-agnostic architecture |
| Causal mask | Prevents decoder from seeing future tokens during training |
| Residual + LayerNorm | Enables deep stacking and stable training |
| Feed-forward network | Position-wise nonlinear transformation |

**Key Takeaway**: The Transformer combines multiple components, each with a clear purpose. Self-attention provides global context and parallelization. Multi-head attention captures diverse relationships. Position encoding solves the order problem. Together, they create an architecture that scales and generalizes remarkably well.

Tomorrow we'll explore how this encoder-decoder design evolved: Why did GPT drop the encoder? Why did BERT drop the decoder? And why did decoder-only models win for language generation?

---

## Appendix A: Deep Dive into Positional Encoding

### A.1 Why Not Just Use Position Numbers?

Why can't we just add the position as a number?

```
Position 0:    embedding + [0]
Position 1:    embedding + [1]
Position 1M:   embedding + [1000000]  ← Explodes!
```

| Problem | Consequence |
|---------|-------------|
| **Numerical instability** | Position 1M drowns out the semantic embedding |
| **Training difficulty** | Huge numerical differences cause unstable gradients |
| **No generalization** | Model never saw position 1M during training |

Sin/cos keeps all values in [-1, 1], matching the scale of embeddings.

### A.2 Step-by-Step Calculation Example

For **position = 3**, **d_model = 8**:

**Formulas:**
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

**Calculate denominator for each dimension pair:**

| i | 2i/d | 10000^(2i/d) | Denominator |
|---|------|--------------|-------------|
| 0 | 0/8 = 0 | 10000^0 | **1** |
| 1 | 2/8 = 0.25 | 10000^0.25 | **10** |
| 2 | 4/8 = 0.5 | 10000^0.5 | **100** |
| 3 | 6/8 = 0.75 | 10000^0.75 | **1000** |

**Why 1, 10, 100, 1000?** Because $10000 = 10^4$:
$$10000^{0.25} = (10^4)^{0.25} = 10^1 = 10$$

**Calculate each dimension:**

```
Dim 0 (i=0, sin): sin(3/1)    = sin(3)     = 0.141
Dim 1 (i=0, cos): cos(3/1)    = cos(3)     = -0.990
Dim 2 (i=1, sin): sin(3/10)   = sin(0.3)   = 0.296
Dim 3 (i=1, cos): cos(3/10)   = cos(0.3)   = 0.955
Dim 4 (i=2, sin): sin(3/100)  = sin(0.03)  = 0.030
Dim 5 (i=2, cos): cos(3/100)  = cos(0.03)  = 0.9995
Dim 6 (i=3, sin): sin(3/1000) = sin(0.003) = 0.003
Dim 7 (i=3, cos): cos(3/1000) = cos(0.003) = 0.9999
```

**Result:**
```
PE(3) = [0.141, -0.990, 0.296, 0.955, 0.030, 0.9995, 0.003, 0.9999]
         sin    cos     sin    cos    sin     cos     sin     cos
         └─i=0─┘       └─i=1─┘       └─i=2─┘         └─i=3─┘
        high freq ←──────────────────────────────→ low freq
```

### A.3 Why Pair Sin and Cos Together?

Each dimension pair (sin, cos) shares the same frequency:

```
Dimensions:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]
             sin   cos   sin   cos   sin   cos   sin   cos
              └─i=0─┘     └─i=1─┘     └─i=2─┘     └─i=3─┘
```

**The magic:** Any relative distance k can be expressed as a rotation matrix!

$$\begin{bmatrix} \sin(pos+k) \\ \cos(pos+k) \end{bmatrix} = \begin{bmatrix} \cos k & \sin k \\ -\sin k & \cos k \end{bmatrix} \begin{bmatrix} \sin(pos) \\ \cos(pos) \end{bmatrix}$$

This means the model can learn "distance = k" as a simple linear transformation, regardless of absolute position.

### A.4 How Does This Encode Relative Distance?

Like a clock:
```
3:00 → 5:00 = rotate 2 hours
8:00 → 10:00 = rotate 2 hours

The "rotate 2 hours" operation is the same, regardless of starting point!
```

The model learns:
> "Distance of 2" is the same transformation whether at position 50→52 or 9998→10000.

### A.5 Can It Handle 1M Positions?

**Yes!** Even though values stay in [-1, 1]:

- Sin/cos output is always bounded, no matter how large the input
- 512 dimensions with different frequencies create a unique "fingerprint" for every position
- Low-frequency dimensions distinguish far positions
- High-frequency dimensions distinguish nearby positions

Like binary numbers: each bit is just 0 or 1, but 32 bits can represent 4 billion values.

### A.6 The Multi-Base Number System Intuition

**Key insight:** Positional encoding is like a continuous, multi-base number system!

**Why multiple frequencies?** Single frequency has collisions:

```
Positions:  1   2   3   4   5
mod 3:      1   2   0   1   2   ← Position 1 and 4 both map to 1!
```

**But add a second frequency:**

```
Positions:  1   2   3   4   5
mod 3:      1   2   0   1   2
mod 5:      1   2   3   4   0

Position 1: (1, 1)
Position 4: (1, 4)  ← Different combination!
```

**This is exactly how number systems work:**

```
Decimal 1234:
  ones digit  = 4      (mod 10)
  tens digit  = 3      (mod 10)  
  hundreds    = 2      (mod 10)
  thousands   = 1      (mod 10)

Positional Encoding 1234:
  dims 0-1 = sin/cos(1234/1)     (high-freq "ones")
  dims 2-3 = sin/cos(1234/10)    (mid-freq "tens")
  dims 4-5 = sin/cos(1234/100)   (low-freq "hundreds")
  dims 6-7 = sin/cos(1234/1000)  (ultra-low "thousands")
```

**Why continuous sin/cos instead of discrete digits?**

| Discrete (0-9) | Continuous sin/cos |
|----------------|-------------------|
| 9→10 is a jump | Smooth transition |
| Not differentiable | Differentiable ✅ |
| Adjacent positions differ wildly | Adjacent positions are similar ✅ |

Neural networks love **smooth** inputs!

> **One sentence:** Positional encoding = continuous multi-base number system, using different frequency sin/cos as different "digit places", smooth and differentiable.

### A.7 The Odometer Analogy

Think of a car's odometer:

```
Odometer: [ones] [tens] [hundreds] [thousands]

Position 0:    0      0      0        0
Position 1:    1      0      0        0     ← ones changed
Position 9:    9      0      0        0
Position 10:   0      1      0        0     ← tens changed
Position 99:   9      9      0        0
Position 100:  0      0      1        0     ← hundreds changed
Position 999:  9      9      9        0
Position 1000: 0      0      0        1     ← thousands changed
```

- **Ones digit = high frequency** (changes every step)
- **Thousands digit = low frequency** (changes every 1000 steps)

Positional encoding works the same way:

| Dimension Type | Role | Analogy |
|----------------|------|---------|
| **High-freq dims** | Distinguish nearby: 50 vs 51 | Odometer ones |
| **Low-freq dims** | Distinguish far: 50 vs 5000 | Odometer thousands |

Any single dimension can't uniquely identify position—but the combination can!

### A.8 Why Not Other Functions?

| Function | Problem |
|----------|---------|
| Linear (pos) | Explodes for long sequences |
| Exponential (e^pos) | Explodes even faster |
| Polynomial (pos²) | Relative distance isn't a simple linear transform |
| **Sin/Cos** | ✅ Bounded, relative distance is linear transform, extrapolates |

### A.9 Modern Alternatives

Sinusoidal isn't the only option—or even the best:

| Method | Year | Used By | Key Feature |
|--------|------|---------|-------------|
| **Sinusoidal** | 2017 | Original Transformer | Fixed, extrapolates |
| **Learned PE** | 2018 | BERT, GPT-2/3 | Trained, slightly better but can't extrapolate |
| **RoPE** | 2021 | LLaMA, Qwen | Rotary encoding, excellent extrapolation |
| **ALiBi** | 2022 | BLOOM | Modifies attention scores directly |

> **Modern LLMs mostly use RoPE** because it combines the extrapolation benefits of sinusoidal with better performance.

### A.10 Python Implementation

```python
import numpy as np

def positional_encoding(max_len, d_model):
    """Generate positional encoding matrix."""
    PE = np.zeros((max_len, d_model))
    
    for pos in range(max_len):
        for i in range(d_model // 2):
            denominator = 10000 ** (2 * i / d_model)
            PE[pos, 2*i] = np.sin(pos / denominator)
            PE[pos, 2*i + 1] = np.cos(pos / denominator)
    
    return PE

# Example: 100 positions, 8 dimensions
PE = positional_encoding(100, 8)
print(f"Position 3 encoding: {PE[3]}")
# [0.141, -0.990, 0.296, 0.955, 0.030, 0.9995, 0.003, 0.9999]
```

---

*Day 4 of 60 | LLM Fundamentals*
*Word count: ~5000 | Reading time: ~22 minutes*
