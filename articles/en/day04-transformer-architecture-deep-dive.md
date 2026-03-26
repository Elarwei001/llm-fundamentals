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

*Day 4 of 60 | LLM Fundamentals*
*Word count: ~4200 | Reading time: ~18 minutes*
