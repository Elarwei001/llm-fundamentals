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

![Figure 4: Timeline of FlashAttention versions, each targeting a new GPU generation](../zh/images/day28/flashattention-evolution-timeline.png)
*Figure 4: FlashAttention has evolved through four generations, each co-designed with the target GPU architecture.*

| Version | Date | Target GPU | Key Innovation | Speedup |
|---------|------|-----------|----------------|---------|
| FlashAttention-1 | Jun 2022 | A100 (Ampere) | IO-aware tiling, online softmax | 2-4x over baseline |
| FlashAttention-2 | Jun 2023 | A100/H100 | Better work partitioning, less non-matmul ops | 2x over FA-1 |
| FlashAttention-3 | Aug 2024 | H100 (Hopper) | Async execution, warp specialization | 1.5-2x over FA-2 |
| FlashAttention-4 | Mar 2026 | B200 (Blackwell) | Asymmetric scaling, CuTe-DSL | 1.3x over cuDNN |

Below, each generation is unpacked in more detail.

### 4.1 FlashAttention-1 (Tri Dao and collaborators, 2022)

FlashAttention-1 introduced two key ideas.

#### Innovation 1: IO-aware tiling

**Principle**: do not construct the full `S = QK^T` and `P = softmax(S)` matrices at once. Instead, slice Q, K, and V into SRAM-sized blocks and compute block by block.

**What problem does it solve?**
- avoids materializing the full `N × N` attention matrix in HBM
- greatly reduces intermediate writeback and rereads
- keeps more work inside fast on-chip SRAM

#### Innovation 2: online softmax

**Principle**: standard softmax wants to see the full row before normalization, but FlashAttention maintains a rolling max and rolling sum so it can process one tile at a time while staying numerically stable.

**What problem does it solve?**
- makes blockwise exact attention mathematically valid
- avoids storing the full score matrix before normalization
- preserves exact attention while removing huge intermediate storage costs

A good summary of FA-1 is:

> **It did not change the definition of attention. It attacked the IO cost of exact attention.**

### 4.2 FlashAttention-2 (a follow-up iteration on the same Tri Dao research line, 2023)

FA-2 did not reinvent the algorithm. It pushed GPU utilization further.

#### Innovation 3: better work partitioning

**Principle**: parallelize more effectively across sequence length and distribute work more evenly across thread blocks.

**What problem does it solve?**
- improves occupancy
- keeps more SMs busy at once
- reduces throughput loss caused by uneven work allocation

#### Innovation 4: reduce the overhead of non-matmul operations

**Principle**: GPUs are best at tensor-core matrix multiplication. Softmax, scaling, masking, indexing, and synchronization are necessary, but they should contribute as little overhead as possible.

**What problem does it solve?**
- reduces the engineering overhead around non-matmul work
- keeps more time on the high-throughput matmul path
- turns theoretical optimization into real end-to-end speedup

### 4.3 FlashAttention-3 (the same research line, optimized for Hopper, 2024)

FlashAttention-3 was designed specifically for NVIDIA Hopper H100 GPUs.

**Hopper** is the GPU architecture generation after Ampere. Its importance here is not just that it is newer. It provides stronger support for async data movement, execution organization, and tensor-core-friendly data paths. In other words, Hopper is not a mysterious new concept, but it is a generation that is especially friendly to high-performance attention kernels.

#### Innovation 5: asynchronous execution

**Principle**: overlap loading the next chunk of data with computing the current chunk.

If this sounds like a classic pipeline idea, that intuition is correct. It belongs to the same family as prefetching, double buffering, and latency hiding.

> **Do not make loading and compute wait in a strict sequence. Overlap them whenever possible.**

What is new here is not the existence of pipelining as an idea, but how that idea is realized inside Hopper-style tile-level FlashAttention execution.

**What problem does it solve?**
- reduces idle time while waiting for data
- hides part of memory latency
- keeps compute units working more continuously

#### Innovation 6: warp specialization

**Principle**: let different warps play different roles instead of making every warp do the same mix of work.

A **warp** is a hardware-scheduled group of 32 threads on NVIDIA GPUs. So warp specialization means, for example:
- some warps focus on data movement
- others focus on computation

**What problem does it solve?**
- reduces role interference inside the kernel
- improves pipelining
- makes better use of Hopper execution behavior

#### Innovation 7: tensor-core path optimization / in-register transpose

**Principle**: use data layouts and in-register transforms that feed Hopper tensor cores more efficiently while avoiding shared-memory bank conflicts.

**What problem does it solve?**
- reduces shared-memory conflicts
- improves tensor-core utilization
- prevents layout inefficiencies from starving the main compute path

### 4.4 FlashAttention-4 (Zadouri et al., March 2026)

FlashAttention-4 faces a new challenge: **asymmetric hardware scaling** on NVIDIA Blackwell B200.

This means the hardware did not scale uniformly.
- **What improved a lot**: tensor-core / matmul throughput. For BF16, peak throughput increased from roughly 1 PFLOPs to about 2.25 PFLOPs.
- **What did not scale nearly as much**: shared-memory bandwidth, SFU throughput for functions like `exp()`, and other supporting components on the non-matmul path.

So Blackwell is not a case where every part of the GPU got proportionally stronger. It is more like the main engine got much faster while some supporting pipelines did not keep up. As a result, bottlenecks shifted away from tensor cores toward non-matmul work and data pathways.

#### Innovation 8: software-emulated exponential

**Principle**: replace some hardware `exp()` calls with polynomial approximation because SFUs did not scale with tensor-core throughput.

**What problem does it solve?**
- prevents SFUs from becoming the new bottleneck
- keeps exponentials from stalling the softmax path
- helps expose the added tensor-core throughput in real workloads

#### Innovation 9: conditional softmax rescaling

**Principle**: online softmax involves many rescaling steps. FA-4 reduces unnecessary rescaling through more selective conditions.

A simple intuition: earlier tiles may have been accumulated under an old maximum value. If a later tile reveals a larger maximum, the old accumulated values must be converted to the new scale before the results can be merged correctly. FA-4 does not eliminate rescaling. It tries to do it only when it is actually needed.

A tiny example helps. Suppose one attention row is seen in two tiles:
- tile 1: `[2, 1]`
- tile 2: `[5, 4]`

When you first see tile 1, the current max is `2`, so you accumulate under the scale "subtract 2":
- `2 -> e^(2-2) = 1`
- `1 -> e^(1-2) = e^(-1)`

So the running sum is `1 + e^(-1)`.

Later, tile 2 reveals that the true max is actually `5`. Now the earlier accumulated values must be converted to the new scale "subtract 5":
- what used to be `1` becomes `e^(2-5) = e^(-3)`
- what used to be `e^(-1)` becomes `e^(1-5) = e^(-4)`

So the previous accumulation must be multiplied by a correction factor `e^(2-5) = e^(-3)`. That conversion step is the rescaling.

**What problem does it solve?**
- reduces extra scalar work in the softmax path
- cuts overhead from necessary but non-matmul operations
- leaves more time for the main operator, namely the tensor-core-dominated matrix multiplication path such as `QK^T` and `PV`

#### Innovation 10: Tensor Memory + 2-CTA MMA mode

**Principle**: use Blackwell's tensor memory features together with a 2-CTA (cooperative thread array) matrix multiplication organization to reduce shared-memory traffic.

> **Terminology box**
>
> - **tensor memory (TMEM)**: according to NVIDIA's official documentation, Blackwell exposes this as a new **`tmem` first-class data locale**. That means it is not just a software reuse of registers or shared memory. A safer interpretation is that it is a new, Tensor-Core / MMA-oriented data locale and hardware support mechanism. NVIDIA's tooling explicitly checks for invalid tensor-memory access, misaligned tensor-memory access, and allocation / relinquish semantics. So TMEM should not be described as just "one more generic cache." It is better understood as part of the data-feeding infrastructure for the main tensor-core path.
> - **CTA (Cooperative Thread Array)**: in CUDA terms, this is very close to a **thread block**, a group of threads that can synchronize and share shared memory.
> - **MMA (Matrix Multiply-Accumulate)**: matrix multiplication plus accumulation, e.g. `C = A × B + C`, the core job tensor cores are built for.
> - **2-CTA MMA**: instead of letting one CTA handle a matrix multiplication block alone, two CTAs cooperate on one MMA pathway, improving data delivery and reducing shared-memory traffic.
> - **Triton**: a system / compiler framework for writing high-performance GPU kernels. "2.7x faster than Triton" means faster than a Triton-based baseline implementation.
> - **cuDNN**: NVIDIA's official deep-learning systems library. "1.3x faster than cuDNN 9.13" means FA-4 outperformed even NVIDIA's own official implementation in that comparison.

**What problem does it solve?**
- relieves shared-memory bandwidth pressure
- better matches Blackwell's new data pathways
- avoids the situation where compute throughput doubles but data feeding still lags behind

FA-4 achieves up to 1613 TFLOPs/s on B200 (71% utilization), 2.7x faster than Triton and 1.3x faster than cuDNN 9.13.

Across all four generations, a helpful summary is:
- **FA-1**: attack the IO cost of exact attention
- **FA-2**: improve GPU parallelism and work partitioning
- **FA-3**: align the kernel more tightly with Hopper execution behavior
- **FA-4**: redesign again for Blackwell's new bottlenecks

---

## 5. Sparse Attention: Skip What Doesn't Matter

FlashAttention computes *exact* attention more efficiently. But what if we don't need to attend to *every* token? Sparse attention takes a different approach: **don't compute attention scores that are zero or negligible anyway**.

#### Intuition: Reading a Book vs Scanning Every Word

When you read a long document, you don't pay equal attention to every word. You focus on the current paragraph, occasionally glance at section headers, and rarely jump to distant pages. Sparse attention mimics this: most tokens only attend to nearby tokens (local window), with a few special tokens (global) that connect everything.

![Figure 5: Six sparse attention patterns showing different sparsity structures](../zh/images/day28/sparse-attention-patterns.png)
*Figure 5: Different sparse attention patterns. Blue cells indicate computed attention; white cells are skipped. Each pattern captures different structural assumptions about which tokens need to interact. Local-window and strided / dilated sparse patterns often look structurally similar to convolutional or dilated-convolution receptive fields. But attention weights are still input-dependent and dynamically computed, unlike the fixed shared weights of a convolution kernel.*

### 5.1 Common Sparse Patterns

First, here is the overview table. Then we unpack each pattern.

| Pattern | Complexity | Best For | Example |
|---------|-----------|----------|---------|
| Local (Sliding Window) | O(N × w) | Local context, code | Mistral, Gemma |
| Strided / Dilated | O(N × s) | Periodic patterns, music | Longformer variant |
| Global + Local | O(N × (w + g)) | Document-level tasks | Longformer, BigBird |
| Block Sparse | O(N²/b²) | Structured data | Block-Sparse Transformer |
| Random Sparse | O(N × r) | Approximate full attention | Sparse Transformer |

Where `w` = window size, `g` = number of global tokens, `b` = block size, and `r` = number of random sampled connections.

#### 5.1.1 Local (sliding-window) attention

**Representative authors / works**: the general idea appeared early in long-sequence Transformers, but in modern LLM practice the most visible deployment is **Sliding Window Attention in Mistral AI (2023)** and later adoption by models such as **Gemma**.

**How it works**: each token attends only to a fixed local neighborhood instead of the full sequence.

**What problem does it solve?**
- reduces cost from `O(N²)` to `O(N × w)`
- preserves the most important short-range dependencies when locality dominates
- works well with KV cache and long-context inference

**Best for**:
- language modeling where local context dominates
- code modeling, where nearby tokens matter a lot
- long-context inference where bounded compute is important

**Limitation**: a single layer cannot directly see faraway tokens. Long-range influence must propagate across multiple layers.

#### 5.1.2 Strided / dilated sparse attention

**Representative authors / works**: this idea is closely related to dilated convolutions and often appears in long-sequence Transformer variants rather than as one universally named flagship model.

**How it works**: instead of looking at a fully contiguous local window, each token samples positions at a fixed stride.

**What problem does it solve?**
- reaches farther distances with fewer edges
- expands the effective receptive field without returning to full attention
- can fit periodic or repeating structures better than a pure contiguous window

**Best for**:
- music, periodic signals, or structured time series
- settings where fixed-interval relationships matter

**Limitation**: if the real dependency pattern is not periodic or regular, a fixed stride may miss important interactions.

#### 5.1.3 Global + local attention

**Representative authors / works**:
- **Longformer** (Iz Beltagy, Matthew E. Peters, Arman Cohan, Allen Institute for AI, 2020)
- **BigBird** (Manzil Zaheer et al., Google Research, 2020)

The shared insight is that not every token needs global visibility, but a small number of important tokens probably should have it.

**How it works**:
- most tokens use local attention
- a few special tokens get global access
- BigBird additionally mixes in random connections

**What problem does it solve?**
- preserves cross-document aggregation while controlling cost
- works better than purely local windows for document-scale understanding
- allows a few global nodes to act as long-range routers

**Best for**:
- long-document classification
- document QA
- multi-paragraph summarization
- NLP tasks that require cross-section integration

**Limitation**: which tokens should be global, and how many, is task-dependent.

#### 5.1.4 Block-sparse attention

**Representative authors / works**:
- **Sparse Transformer** (Rewon Child et al., OpenAI, 2019) strongly emphasized structured sparsity
- later system work often prefers block-sparse designs because hardware likes regularity

**How it works**: instead of deciding sparsity token by token, partition the attention matrix into blocks and keep or drop whole blocks.

**What problem does it solve?**
- is easier to implement efficiently than highly irregular pointwise sparsity
- aligns better with GPU / accelerator data movement patterns
- keeps useful structure while remaining hardware-friendly

**Best for**:
- structured data
- image patches or video patches
- system designs that need sparsity to align with hardware execution patterns

**Limitation**: the more regular the structure, the easier the implementation, but potentially the less expressive the pattern.

#### 5.1.5 Random sparse attention

**Representative authors / works**:
- **Sparse Transformer** (Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever, OpenAI, 2019)
- **BigBird** also includes random edges as part of its sparse graph

**How it works**: beyond local connections, each token gets a small number of random long-range links.

From a graph-theoretic perspective, this resembles a **small-world network**: most edges are local, but a few long-range shortcuts dramatically reduce average path length. In attention terms, that means you do not necessarily need full all-to-all connectivity for information to travel efficiently across a long sequence.

**What problem does it solve?**
- retains some global communication ability at relatively low cost
- reduces the bottleneck of purely local connectivity
- improves connectivity through shortcut-like long-range edges

**Best for**:
- very long sequence modeling
- settings where we want some global shortcuts without paying full-attention cost

**Limitation**: random patterns can be theoretically attractive but are often less interpretable and less implementation-friendly than regular local or block-sparse schemes.

### 5.2 The Sliding Window in Practice

The most popular sparse pattern today is the **sliding window attention** used by models like Mistral and Gemma. Each token only attends to the previous w tokens (typically w = 4096 or 8192):

$$
A_{ij} = \begin{cases} \text{softmax}(Q_i K_j^T / \sqrt{d}) & \text{if } |i - j| \leq w \\ 0 & \text{otherwise} \end{cases}
$$

With multiple layers, information still propagates across the full sequence. A token at position 0 can influence position 1000 through L hops, where each hop extends the effective receptive field by w tokens. After L layers, the receptive field is L × w tokens.

### 5.3 FlexAttention: Making Sparse Attention Easy

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

## 6. Combining FlashAttention and Sparse Attention

These two approaches are complementary:

| Approach | What it optimizes | Approximation? |
|----------|-------------------|----------------|
| FlashAttention | Memory bandwidth (exact computation) | No — bit-exact |
| Sparse Attention | Number of computed elements | Yes — drops some connections |
| Both together | Memory bandwidth × sparsity ratio | Sparse approximation, computed efficiently |

Modern LLMs typically use both: FlashAttention as the backend kernel, with sparse masks (like sliding window) applied on top. The combination is powerful — if your sparsity pattern drops 80% of entries, and FlashAttention is 3x faster for each entry it *does* compute, you get an overall ~15x speedup.

---

## 7. Common Misconceptions

### ❌ "FlashAttention approximates attention for speed"

No. FlashAttention produces **exact, bit-for-bit identical** results to standard attention. The speedup comes entirely from reducing memory reads/writes (IO), not from approximating the computation. You can verify this by comparing outputs — they match to floating-point precision.

### ❌ "Sparse attention always hurts quality"

Not necessarily. Research has shown that attention matrices in trained LLMs are naturally sparse — most attention weights are near zero. Sparse attention patterns often align with this natural sparsity, so the quality impact can be minimal. Models like Mistral (sliding window) and Longformer (global + local) achieve strong results with sparse patterns.

### ❌ "FlashAttention-4 makes sparse attention unnecessary"

FlashAttention-4 speeds up *each attention computation*, but the overall cost is still O(N²) for full attention. For very long sequences (100K+ tokens), combining FlashAttention with sparse patterns is still essential. They solve different parts of the problem.

---

## 8. Frontier: What's New (2025-2026)

1. **FlashAttention-4** (Zadouri et al., March 2026) — Co-designed for NVIDIA Blackwell B200 with asymmetric hardware scaling. Achieves 71% utilization (1613 TFLOPs/s) by replacing hardware `exp()` with software emulation and leveraging new tensor memory features. ([Paper](https://arxiv.org/abs/2603.05451), [Together AI blog](https://www.together.ai/blog/flashattention-4))

2. **FlexAttention + FlashAttention-4 backend** (PyTorch team, March 2026) — FlexAttention now uses FA-4 as its backend, enabling custom sparse patterns with near-optimal performance. 1.2x speedup over previous backends for custom attention variants. ([PyTorch blog](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/))

3. **Long-Context Generalization with Sparse Attention** (Peters et al., accepted at ICLR 2026) — Uses α-entmax to produce *naturally sparse* attention distributions (exact zeros, not near-zeros), improving length generalization without hand-designed sparsity patterns. ([Paper](https://arxiv.org/abs/2506.16640))

   This one is worth slowing down for, because it differs from Longformer, BigBird, and sliding-window designs. Those approaches mostly start by **hand-specifying a sparse topology**, such as "each token can only see its nearby 4096 tokens" or "a few special tokens are global." This paper instead explores a different idea:

   > **Do not hand-design who can attend to whom. Let the attention distribution learn sparsity by itself.**

   The main tool is **α-entmax**, which you can think of as a sparse alternative to softmax:
   - **softmax** gives almost every position a positive weight, even if many are tiny
   - **α-entmax** can push many positions to **exactly zero**

   That difference matters in long-context settings. "Tiny but nonzero" still means the model is mathematically attending a little bit everywhere. "Exactly zero" means the model has learned that some positions should not be attended at all.

   Why could this help long-context generalization? One intuition is that when context length grows from, say, 4K during training to 32K or 64K at test time, a model that keeps spreading some probability mass almost everywhere may become diffuse and lose focus. A model that learns to collapse most irrelevant positions to zero can keep a sharper, more selective attention pattern even as the sequence gets longer.

   So the real question this line of work asks is not just:
   - how do we manually design a good sparse graph?

   but rather:
   - **can the model itself learn to become sparse and selective in longer contexts?**

   Put differently, the key idea is: **replace softmax with a sparse normalization function so the model learns during training which attention weights should truly become zero, instead of applying thresholding or pruning only after training.**

   This is promising, but it also raises practical questions:
   - how stable is training with sparse distributions?
   - can the learned sparsity pattern actually be exploited efficiently by kernels?
   - will content-adaptive sparsity translate into real inference speedups, or only into a nicer analytical story?

4. **Efficient Attention Mechanisms Survey** (July 2025) — Comprehensive survey categorizing the explosion of attention variants: hardware-aware (FlashAttention), sparse, linear, and hybrid approaches. ([Paper](https://arxiv.org/abs/2507.19595))

---

## 9. Further Reading

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

## 10. Reflection Questions

1. Why is memory bandwidth (not compute) the bottleneck for attention? What does this tell us about where GPU hardware is heading?
2. If FlashAttention gives exact results, what prevents us from just using it everywhere? (Hint: think about what FlashAttention *doesn't* optimize.)
3. Sparse attention assumes we can predict which tokens matter *before* computing attention. Is this assumption valid? When might it break down?

---

## 11. Summary

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
*Word count: ~6200 | Reading time: ~30-38 minutes*
