# Day 28: FlashAttention & Sparse Attention

> **Core Question**: Why is attention the bottleneck in Transformers, and how do FlashAttention and sparse attention patterns solve the memory wall?

---

## Opening

Imagine you're a chef preparing a banquet for 10,000 guests. You have a massive pantry (HBM, High Bandwidth Memory) with tons of ingredients, but a tiny kitchen counter (SRAM) where you can only work with a few items at a time. The naive approach would be to bring *all 10,000 plates' worth of ingredients* to your counter at once — impossible. Instead, you work in batches: prepare 50 plates at a time, keep running totals, and never try to hold everything simultaneously.

This is exactly the problem attention faces in modern LLMs. The attention matrix grows as N² for a sequence of N tokens. For a 128K-token context, that's a 128K × 128K matrix — roughly 64 GB of memory just for one attention layer. FlashAttention and sparse attention are two complementary strategies that tackle this problem from different angles: FlashAttention computes *exact* attention more efficiently, while sparse attention computes *approximate* attention with fewer operations.

Today we'll understand both — and why they matter for every LLM you use.

---

## 1. The Memory Bandwidth Problem

#### Intuition: The Data Moving Problem

Think of attention like a factory floor. The workers (compute units) are fast — they can multiply matrices all day. But the supplies (data) come from a warehouse across town (HBM, High Bandwidth Memory), and loading/unloading is the real bottleneck. It doesn't matter how fast your workers are if they spend most of their time waiting for materials to arrive.

![Figure 1: GPU memory hierarchy showing the bandwidth vs capacity gap](../zh/images/day28/attention-memory-bottleneck.png)
*Figure 1: GPU memory has a fundamental tension — HBM is large but slow, SRAM is fast but tiny. Attention is bottlenecked by the data movement between them. Note: the SRAM capacity bar may look almost like zero, not because SRAM has no capacity, but because it is usually on the MB scale; when plotted against tens or hundreds of GB of HBM on the same axis, it gets visually compressed.*

Standard attention computes:

$$
\begin{aligned}
S &= Q K^T \quad &(N \times N \text{ score matrix}) \\
P &= \text{softmax}(S) \quad &(N \times N \text{ attention weights}) \\
O &= P V \quad &(N \times d \text{ output})
\end{aligned}
$$

The intermediate matrices S and P are both N × N. For N = 128K tokens with FP16 (2 bytes per element), each matrix is 128K × 128K × 2 = **32 GB**. With multiple attention heads and layers, this quickly exhausts HBM.

But here's the key insight from the FlashAttention paper (Dao et al., 2022): **the arithmetic of attention is not the bottleneck, memory bandwidth is.** The GPU's tensor cores can do the math much faster than data can be moved in and out of HBM. The bottleneck is *IO*, not *FLOPs*.

Before reading the table below line by line, focus on the qualitative point:

- **Standard attention** repeatedly writes large intermediate results to HBM and reads them back.
- **FlashAttention** tries to keep intermediate work inside much faster on-chip SRAM, and only writes back what is truly needed.

So you can read the table as a preview: FlashAttention does **not** change the definition of attention. It changes the **implementation path**, reducing memory use and HBM traffic.

| Metric | Standard Attention | FlashAttention |
|--------|--------------------|----------------|
| Memory | O(N²) | O(N) |
| HBM reads | O(N²d) | O(N²d / M) where M = SRAM size |
| HBM writes | O(N²) | O(Nd) |
| Math | Exact | Exact (same result) |

FlashAttention does not approximate. It preserves the final definition of attention and produces results consistent with standard attention. The savings come entirely from smarter data movement.

---

## 2. How FlashAttention Works

### 2.1 Online Softmax: The Key Trick

#### Intuition: Running Average Instead of Re-reading Everything

Imagine you're calculating the class average for 1,000 students. The naive approach: read all 1,000 grades, sum them, divide. The smart approach: maintain a running sum and count as you go. You never need to hold all 1,000 grades at once.

Standard softmax needs to see all values before normalizing:

$$
\text{softmax}(s_i) = \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

This requires knowing the denominator for all j — meaning you can't compute it tile-by-tile. FlashAttention solves this with **online softmax** (Milakov & Gimelshein, 2018): it maintains running statistics (the current max and sum) and rescales previously computed results when new tiles arrive.

### 2.2 Tiling: Never Materialize the Full Matrix

![Figure 2: FlashAttention loads tiles of Q, K, V from HBM into SRAM, computes partial attention in SRAM, and writes only the output back](../zh/images/day28/flashattention-tiling-memory-v2.png)
*Figure 2: FlashAttention's tiling strategy. Here, a tile just means a small block cut out of a larger matrix. Instead of materializing the full N×N attention matrix in HBM, tiles of Q, K, V are loaded into fast SRAM, partial attention is computed, and results are accumulated using online softmax.*

The algorithm works as follows:
1. Load a tile of Q (rows i₁ to i₂) and a tile of K (rows j₁ to j₂) into SRAM
2. Compute the partial attention scores for this block
3. Update the running output O using online softmax rescaling
4. Move to the next K tile, repeat until all K tiles are processed
5. Move to the next Q tile, repeat

The result: instead of reading/writing an N × N matrix from HBM, you only read Q, K, V (each N × d) and write O (N × d). Memory usage drops from O(N²) to O(N). Tile-level computation here simply means organizing computation at the level of these small blocks rather than processing the entire matrix at once.

The most important point, and the easiest one to miss, is this: **FlashAttention does not save IO mainly because Q, K, and V suddenly become much smaller. It saves IO because it no longer writes the huge intermediate matrices `S = QK^T` and `P = softmax(S)` back to HBM and then reads them again later.**

In other words:
- **Standard attention** often materializes full `S` and `P`, creating large HBM writes and later reads.
- **FlashAttention** keeps intermediate work on-chip, maintaining only rolling max, rolling sum, and the running output accumulator, then writes back only what is actually needed.

So the real savings come from avoiding the materialization, writeback, and rereading of large intermediate results.

### 2.3 Code Comparison: Where Standard Attention Writes Back to HBM, and Where FlashAttention Saves It

First, here is a more standard, intuitive formulation. The issue is not mathematical correctness. The issue is that it explicitly materializes the big intermediate matrices:

```python
import torch

def standard_attention_forward(Q, K, V):
    """
    Intuitive standard attention implementation.
    The point here is not peak efficiency.
    The point is to show which intermediate states become explicit.
    """
    d = Q.shape[-1]

    S = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    # Full score matrix S = QK^T is explicitly created here.

    P = torch.softmax(S, dim=-1)
    # Full probability matrix P = softmax(S) is explicitly created here.

    O = torch.matmul(P, V)
    return O
```

The key thing to notice is the explicit existence of two full N×N matrices:
- `S = QK^T`
- `P = softmax(S)`

That is exactly why long-sequence attention becomes so expensive in memory traffic.

Now compare this with FlashAttention:

```python
import torch

def flash_attention_forward(Q, K, V, block_size=64):
    """
    Simplified FlashAttention forward pass.
    Q, K, V: (batch, heads, seq_len, head_dim)
    Produces exact same result as torch.nn.functional.scaled_dot_product_attention
    """
    B, H, N, d = Q.shape
    O = torch.zeros_like(Q)           # Output accumulator
    l = torch.zeros(B, H, N, 1, device=Q.device)  # Running sum of exp
    m = torch.full((B, H, N, 1), float('-inf'), device=Q.device)  # Running max
    
    for j_start in range(0, N, block_size):
        j_end = min(j_start + block_size, N)
        K_block = K[:, :, j_start:j_end, :]   # Load K tile
        V_block = V[:, :, j_start:j_end, :]   # Load V tile
        
        for i_start in range(0, N, block_size):
            i_end = min(i_start + block_size, N)
            Q_block = Q[:, :, i_start:i_end, :]  # Load Q tile
            
            # Compute partial attention scores for this block
            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) / (d ** 0.5)
            
            # Online softmax: update running max and rescale
            m_new = torch.max(m[:, :, i_start:i_end, :], 
                              S_block.max(dim=-1, keepdim=True).values)
            
            # Rescale previous accumulated values
            exp_diff = torch.exp(m[:, :, i_start:i_end, :] - m_new)
            exp_S = torch.exp(S_block - m_new)
            
            # Update running sum and output
            l[:, :, i_start:i_end, :] = l[:, :, i_start:i_end, :] * exp_diff + exp_S.sum(dim=-1, keepdim=True)
            O[:, :, i_start:i_end, :] = O[:, :, i_start:i_end, :] * exp_diff + torch.matmul(exp_S, V_block)
            
            m[:, :, i_start:i_end, :] = m_new
    
    # Final normalization
    O = O / l
    return O
```

This simplified version shows the core idea. In the FlashAttention version, `S_block` is only a local temporary block, not a full materialized N×N matrix. Likewise, the local exponentiated values are consumed immediately instead of becoming a full `P` matrix written to HBM.

---

## 3. Hardware Primer: Why FlashAttention Is So "Close to the GPU"

If terms like the following still feel fuzzy, many of the later optimizations will read like magic:
- SM
- warp
- tensor core
- register
- shared memory / SRAM
- HBM
- SFU

So before the version-by-version evolution, here is a compact NVIDIA GPU primer.

![Figure 3: NVIDIA GPU hardware sketch showing SMs, warps, tensor cores, shared memory, and HBM](../zh/images/day28/nvidia-gpu-hardware-primer-v2.png)
*Figure 3: You can think of an NVIDIA GPU as a collection of SMs (compute workshops). Inside each SM are tensor cores, CUDA cores, registers, shared memory, warp schedulers, and related execution units. Data comes in from HBM. FlashAttention-like optimizations try to keep intermediate results in registers / shared memory rather than repeatedly writing large intermediate matrices back to HBM.*

### 3.1 How is a GPU organized?

A useful mental model is:
- **GPU**: the full factory
- **SM (Streaming Multiprocessor)**: one compute workshop
- **warp**: a 32-thread team inside a workshop
- **thread**: one worker

Many kernel optimizations boil down to two questions:
1. How do we keep more SMs busy at once?
2. How do we keep the warps inside each SM waiting less and computing more?

### 3.2 SM: the compute workshop inside a GPU

An **SM** is a relatively self-contained execution unit inside the GPU. It typically contains:
- **CUDA cores** for more general scalar/vector work
- **tensor cores** for high-throughput matrix multiplication
- **registers** for the fastest thread-local storage
- **shared memory** for fast block-local collaboration
- **warp schedulers** for deciding which warp runs next
- and, in newer generations, stronger support for async movement and execution organization

The point is not to memorize the parts. The point is to realize that many optimizations are really about competing for compute, on-chip storage, and scheduling opportunities inside the SM.

### 3.3 warp: the basic scheduling group

On NVIDIA GPUs, a **warp** is typically **32 threads** scheduled together as a unit. So the GPU does not really think in terms of fully independent threads. It thinks in terms of warp-sized work.

That is why **warp specialization** makes sense:
- some warps can focus on loading data
- others can focus on compute
- and together they create a better pipeline

### 3.4 One SM and many warps: how GPUs hide latency

An SM is not paired with just one warp. Usually, it has **many resident warps**.

> **One SM : many warps**

This is central to GPU latency hiding. If one warp is waiting for data, the SM does not sit idle. The **warp scheduler** can choose another ready warp instead.

#### Where does inactive warp state live?

Warp state usually stays resident on the SM:
- **register state** sits in the SM's register file
- **program counters and execution state** are maintained in hardware
- **shared data** lives in block-visible shared memory

That is why switching between warps is so cheap compared with heavyweight CPU-style context switching.

#### How many warps can one SM host?

Not infinitely many. It depends on:
- register usage
- shared memory usage
- block configuration
- architectural limits

This is why **occupancy** matters. If each warp or block consumes too many resources, the SM can host fewer warps, making it harder to hide latency.

### 3.5 tensor core vs CUDA core: why matmul matters so much

At a high level:
- **CUDA cores** are more general-purpose
- **tensor cores** are specialized for matrix multiply-accumulate work

This is why FlashAttention repeatedly tries to keep the matrix multiplication path well fed. In modern training and inference, large matmuls are often the most valuable, highest-throughput path on the GPU.

### 3.6 registers, shared memory, and HBM: three very different storage layers

#### registers
- fastest
- closest to compute
- thread-private
- tiny

#### shared memory / SRAM
- very fast
- larger than registers, but still small
- shared inside a thread block
- ideal for tiles, partial accumulators, and temporary collaborative state
- crucially, it is **local to each SM**, not a GPU-wide shared pool; communication across SMs typically goes through higher-level cache or global memory

#### HBM (High Bandwidth Memory)
- much larger
- high bandwidth, but still much slower than on-chip storage
- repeated reads/writes of large intermediate matrices quickly become the bottleneck

A large fraction of FlashAttention is really trying to do one thing:

> **keep intermediate work in registers and shared memory, and avoid sending it back to HBM unless necessary.**

### 3.7 SFU: why even `exp()` can become a bottleneck

The **SFU (Special Function Unit)** handles functions like exponentials, logarithms, and trigonometric operations.

Most of the time people do not think about it. But once the main matmul path becomes extremely optimized, something as small as the throughput of `exp()` can emerge as a real bottleneck.

### 3.8 What does "hardware-aware" really mean?

Now the phrase becomes concrete. It means asking:
- how should work be distributed across **SMs**?
- how should **warps** specialize?
- how do we keep **tensor cores** busy?
- which intermediate results should stay in **registers / shared memory**?
- which results become expensive the moment they go back to **HBM**?
- could even something like `exp()` become limited by **SFUs**?

The later FlashAttention generations are best understood as repeated attempts to rewrite the same attention kernel so that it matches what that GPU generation likes best.

---

## 4. FlashAttention Evolution: From v1 to v4

![Figure 3: Timeline of FlashAttention versions, each targeting a new GPU generation](../zh/images/day28/flashattention-evolution-timeline.png)
*Figure 3: FlashAttention has evolved through four generations, each co-designed with the target GPU architecture.*

| Version | Date | Target GPU | Key Innovation | Speedup |
|---------|------|-----------|----------------|---------|
| FlashAttention-1 | Jun 2022 | A100 (Ampere) | IO-aware tiling, online softmax | 2-4x over baseline |
| FlashAttention-2 | Jun 2023 | A100/H100 | Better work partitioning, less non-matmul ops | 2x over FA-1 |
| FlashAttention-3 | Aug 2024 | H100 (Hopper) | Async execution, warp specialization | 1.5-2x over FA-2 |
| FlashAttention-4 | Mar 2026 | B200 (Blackwell) | Asymmetric scaling, CuTe-DSL | 1.3x over cuDNN |

**FlashAttention-2** (Dao, 2023) reduced non-matmul FLOPs by parallelizing across the sequence length dimension and assigning each thread block to a single attention head. The key insight: matmul operations should use the GPU's tensor cores, and everything else should be minimized.

**FlashAttention-3** (Dao et al., 2024) was specifically designed for NVIDIA's Hopper H100 GPUs. It exploited three hardware features:
- **Asynchronous execution**: overlap softmax computation with data loading
- **Warp specialization**: dedicate some warps to data loading, others to computation
- **Tensor core optimizations**: use in-register transpose to avoid shared memory bank conflicts

**FlashAttention-4** (Zadouri et al., March 2026) tackles a new challenge: **asymmetric hardware scaling** on NVIDIA's Blackwell B200 GPUs. On Blackwell, tensor core throughput doubled (1 → 2.25 PFLOPs for BF16), but other components like shared memory bandwidth and exponential function units stayed the same. This means the old tricks from FA-3 don't work — the bottleneck shifted from tensor cores to non-matmul operations.

FA-4's three key techniques:
1. **Software-emulated exponential**: Replace hardware `exp()` with polynomial approximation, because SFU (Special Function Unit) count didn't scale with tensor cores
2. **Conditional softmax rescaling**: Reduce the number of rescaling operations in the online softmax
3. **Tensor Memory + 2-CTA MMA mode**: Use Blackwell's new tensor memory feature to reduce shared memory traffic

FA-4 achieves up to 1613 TFLOPs/s on B200 (71% utilization) — a 2.7x speedup over Triton and 1.3x over cuDNN 9.13.

---

## 4. Sparse Attention: Skip What Doesn't Matter

FlashAttention computes *exact* attention more efficiently. But what if we don't need to attend to *every* token? Sparse attention takes a different approach: **don't compute attention scores that are zero or negligible anyway**.

#### Intuition: Reading a Book vs Scanning Every Word

When you read a long document, you don't pay equal attention to every word. You focus on the current paragraph, occasionally glance at section headers, and rarely jump to distant pages. Sparse attention mimics this: most tokens only attend to nearby tokens (local window), with a few special tokens (global) that connect everything.

![Figure 4: Six sparse attention patterns showing different sparsity structures](../zh/images/day28/sparse-attention-patterns.png)
*Figure 4: Different sparse attention patterns. Blue cells indicate computed attention; white cells are skipped. Each pattern captures different structural assumptions about which tokens need to interact.*

### 4.1 Common Sparse Patterns

| Pattern | Complexity | Best For | Example |
|---------|-----------|----------|---------|
| Local (Sliding Window) | O(N × w) | Local context, code | Mistral, Gemma |
| Strided / Dilated | O(N × s) | Periodic patterns, music | Longformer variant |
| Global + Local | O(N × (w + g)) | Document-level tasks | Longformer, BigBird |
| Block Sparse | O(N²/b²) | Structured data | Block-Sparse Transformer |
| Random Sparse | O(N × r) | Approximate full attention | Sparse Transformer |

Where w = window size, g = number of global tokens, b = block size, r = random sample count.

### 4.2 The Sliding Window in Practice

The most popular sparse pattern today is the **sliding window attention** used by models like Mistral and Gemma. Each token only attends to the previous w tokens (typically w = 4096 or 8192):

$$
A_{ij} = \begin{cases} \text{softmax}(Q_i K_j^T / \sqrt{d}) & \text{if } |i - j| \leq w \\ 0 & \text{otherwise} \end{cases}
$$

With multiple layers, information still propagates across the full sequence. A token at position 0 can influence position 1000 through L hops, where each hop extends the effective receptive field by w tokens. After L layers, the receptive field is L × w tokens.

### 4.3 FlexAttention: Making Sparse Attention Easy

One challenge with sparse attention is that implementing custom patterns requires writing low-level CUDA kernels — extremely difficult and error-prone. **FlexAttention** (PyTorch team, 2024) solves this by letting users define arbitrary attention modifications in pure PyTorch, and it automatically compiles them to efficient kernels.

As of March 2026, FlexAttention integrates with the FlashAttention-4 backend, achieving 1.2x speedup over previous backends for custom attention variants. This means you can write:

```python
from torch.nn.attention.flex_attention import flex_attention

def sliding_window_mask(score, b, h, q_idx, kv_idx):
    return (q_idx - kv_idx) < 512  # Window of 512 tokens

# This automatically gets compiled to an efficient FlashAttention kernel
output = flex_attention(Q, K, V, score_mod=sliding_window_mask)
```

FlexAttention handles the block-sparse optimization automatically — if your mask removes 80% of attention weights, you get roughly 5x speedup without writing any CUDA code.

---

## 5. Combining FlashAttention and Sparse Attention

These two approaches are complementary:

| Approach | What it optimizes | Approximation? |
|----------|-------------------|----------------|
| FlashAttention | Memory bandwidth (exact computation) | No — bit-exact |
| Sparse Attention | Number of computed elements | Yes — drops some connections |
| Both together | Memory bandwidth × sparsity ratio | Sparse approximation, computed efficiently |

Modern LLMs typically use both: FlashAttention as the backend kernel, with sparse masks (like sliding window) applied on top. The combination is powerful — if your sparsity pattern drops 80% of entries, and FlashAttention is 3x faster for each entry it *does* compute, you get an overall ~15x speedup.

---

## 6. Common Misconceptions

### ❌ "FlashAttention approximates attention for speed"

No. FlashAttention produces **exact, bit-for-bit identical** results to standard attention. The speedup comes entirely from reducing memory reads/writes (IO), not from approximating the computation. You can verify this by comparing outputs — they match to floating-point precision.

### ❌ "Sparse attention always hurts quality"

Not necessarily. Research has shown that attention matrices in trained LLMs are naturally sparse — most attention weights are near zero. Sparse attention patterns often align with this natural sparsity, so the quality impact can be minimal. Models like Mistral (sliding window) and Longformer (global + local) achieve strong results with sparse patterns.

### ❌ "FlashAttention-4 makes sparse attention unnecessary"

FlashAttention-4 speeds up *each attention computation*, but the overall cost is still O(N²) for full attention. For very long sequences (100K+ tokens), combining FlashAttention with sparse patterns is still essential. They solve different parts of the problem.

---

## 7. Frontier: What's New (2025-2026)

1. **FlashAttention-4** (Zadouri et al., March 2026) — Co-designed for NVIDIA Blackwell B200 with asymmetric hardware scaling. Achieves 71% utilization (1613 TFLOPs/s) by replacing hardware `exp()` with software emulation and leveraging new tensor memory features. ([Paper](https://arxiv.org/abs/2603.05451), [Together AI blog](https://www.together.ai/blog/flashattention-4))

2. **FlexAttention + FlashAttention-4 backend** (PyTorch team, March 2026) — FlexAttention now uses FA-4 as its backend, enabling custom sparse patterns with near-optimal performance. 1.2x speedup over previous backends for custom attention variants. ([PyTorch blog](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/))

3. **Long-Context Generalization with Sparse Attention** (Peters et al., accepted at ICLR 2026) — Uses α-entmax to produce *naturally sparse* attention distributions (exact zeros, not near-zeros), improving length generalization without hand-designed sparsity patterns. ([Paper](https://arxiv.org/abs/2506.16640))

4. **Efficient Attention Mechanisms Survey** (July 2025) — Comprehensive survey categorizing the explosion of attention variants: hardware-aware (FlashAttention), sparse, linear, and hybrid approaches. ([Paper](https://arxiv.org/abs/2507.19595))

---

## 8. Further Reading

### Foundational
1. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — The original FlashAttention paper (Dao et al., 2022)
2. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023

### Sparse Attention
3. [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — Child et al., 2019. The original sparse attention paper
4. [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Beltagy et al., 2020. Global + local attention

### Hardware-Aware
5. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Dao et al., 2024
6. [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451) — Zadouri et al., 2026

### Tools
7. [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention](https://pytorch.org/blog/flexattention/) — PyTorch, 2024

---

## Reflection Questions

1. Why is memory bandwidth (not compute) the bottleneck for attention? What does this tell us about where GPU hardware is heading?
2. If FlashAttention gives exact results, what prevents us from just using it everywhere? (Hint: think about what FlashAttention *doesn't* optimize.)
3. Sparse attention assumes we can predict which tokens matter *before* computing attention. Is this assumption valid? When might it break down?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| Memory bandwidth bottleneck | Attention is IO-bound, not compute-bound — data movement is the real cost |
| FlashAttention tiling | Load tiles of Q, K, V into SRAM, compute partial attention, accumulate with online softmax |
| Online softmax | Maintain running max and sum to normalize without seeing all values at once |
| O(N²) → O(N) memory | FlashAttention never materializes the full N×N attention matrix in HBM |
| Sparse attention | Skip computing attention for token pairs that are likely irrelevant |
| Sliding window | Each token only attends to nearby tokens; information propagates through layers |
| FlexAttention | PyTorch API for custom attention patterns with automatic kernel optimization |
| FlashAttention-4 | Blackwell-optimized with software `exp()` and tensor memory for asymmetric scaling |

**Key Takeaway**: The attention bottleneck is a *data movement* problem, not a *computation* problem. FlashAttention solves it by working in tiles within fast SRAM, while sparse attention reduces the number of computations needed. Together, they enable the long-context LLMs we use today — and each new GPU generation requires rethinking how to map attention onto changing hardware characteristics.

---

*Day 28 of 60 | LLM Fundamentals*
*Word count: ~4300 | Reading time: ~22-28 minutes*
