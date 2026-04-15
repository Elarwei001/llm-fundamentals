# Day 17: KV Cache

> **Core Question**: Why can a large language model generate one token at a time without recomputing the entire attention history from scratch, and what price does that speedup impose?

---

## Opening

If you have ever watched a chatbot answer in real time, the experience feels smooth, almost conversational. Under the hood, though, autoregressive generation is an awkward workload. The model emits one token, then another, then another. At each step, the new token should attend to the entire prefix. If we literally reran the full Transformer computation over all previous tokens every time, inference would become painfully wasteful.

That waste is exactly what the **KV cache** avoids. The idea is simple: when a token has already passed through self-attention once, the model has already computed its **key** and **value** vectors. Those tensors do not change during inference, so instead of recomputing them at every new decoding step, we store them and reuse them. The result is one of the most important practical tricks in modern LLM serving.

Think of it like taking notes during a long meeting. Without notes, every time someone asks a question you would replay the entire meeting in your head from the beginning. With notes, you keep the old discussion on paper and only process the new remark. The meeting is still long, but the next response becomes much cheaper.

KV cache sounds like an implementation detail, but it is really a bridge between theory and product reality. It explains why long-context inference is expensive, why memory bandwidth matters so much, why batching is hard, and why systems like vLLM introduced **paged attention**. In this article, we will unpack the mechanism carefully, quantify the trade-offs, and connect the concept to real serving systems.

---

## 1. Why naïve autoregressive decoding is wasteful

The probability model is still the usual next-token objective:

$$
P(x_{1:T}) = \prod_{t=1}^{T} P(x_t \mid x_{1:t-1}).
$$

At inference time, generation proceeds step by step. Suppose the prompt has length $n$, and we are generating token $n+1$, then $n+2$, and so on. In a decoder-only Transformer, each new step runs self-attention over the prefix seen so far.

In a single attention head, we compute

> **What is $X$ here?** $X$ is the **input matrix** containing embeddings for all tokens processed so far. If we have processed $n$ tokens, each with embedding dimension $d$, then $X$ is an $n \times d$ matrix — each row is one token's embedding.
>
> The three projections produce:
> - **$Q = XW_Q$** — Query matrix ("what am I looking for?")
> - **$K = XW_K$** — Key matrix ("what information can I provide?")
> - **$V = XW_V$** — Value matrix ("what is my actual content?")
>
> **Example:** Prompt = "I love cats" (3 tokens). $X$ is $3 \times d$. After projection, $Q, K, V$ are each $3 \times d_k$.
>
> **The waste:** When generating token 4 ("are"), $X$ becomes $4 \times d$, and we recompute ALL four tokens' $Q, K, V$. But tokens 1-3's $K$ and $V$ were already computed last step — that's redundant. KV cache eliminates this duplication.

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V,
$$

and then

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
$$

where $M$ is the causal mask.

> **What is $d_k$?** $d_k$ is the dimension of the Key (and Query) vector in each attention head. It is computed as: $d_k = d / h$, where $d$ is the model hidden dimension and $h$ is the number of attention heads.
>
> **Example:** LLaMA-7B has $d = 4096$, $h = 32$, so $d_k = 4096 / 32 = 128$. Each head projects Q, K, V into 128-dimensional space.
>
> **Why divide by $\sqrt{d_k}$?** This is the "scaled" in Scaled Dot-Product Attention. When $d_k$ is large, the dot product $QK^T$ produces large values, which pushes softmax into regions with near-zero gradients (gradient vanishing). Dividing by $\sqrt{d_k}$ normalizes the variance: if $d_k = 128$, random vectors' dot product has variance ~128, and dividing by $\sqrt{128} pprox 11.3$ brings variance back to ~1. This keeps softmax healthy and gradients flowing.

During training, computing all queries, keys, and values together is natural because the whole sequence is available. During decoding, however, token $t$ has already had its $K_t$ and $V_t$ computed in previous steps. Recomputing them again for steps $t+1$, $t+2$, and $t+3$ adds work but no new information.

![Figure 1: Decoding cost with and without cache](../zh/images/day17/prefill-vs-decode-cost.png)
*Caption: Without caching, each decoding step effectively repeats attention work over the past. With caching, the model mostly pays for the new token rather than the full history.*

This is the core inefficiency. If we ignore caching, the total attention work across decoding grows roughly quadratically with generated length, because step 1 attends over 1 token, step 2 attends over 2, step 3 over 3, and so on. The model is not learning anything new from recomputing the past. It is just forgetting that it already did the work.

---

## 2. The basic idea of KV cache

**The key observation is that previous keys and values are fixed during inference.** Once token $i$ has been projected into $K_i$ and $V_i$, those tensors can be stored and reused for future steps.

At decode step $t$, instead of recomputing all projections for tokens $1 \ldots t$, we do this:

1. Compute the new hidden state for token $t$.
2. Project only that token to get its new query, key, and value.
3. Append the new key and value to the cache.
4. Attend using the new query against all cached keys and values.

So the cached state at one layer looks like

$$
K_{1:t} = [K_1, K_2, \ldots, K_t], \qquad V_{1:t} = [V_1, V_2, \ldots, V_t].
$$

Then the next token uses

$$
\text{Attention}(Q_{t+1}, K_{1:t+1}, V_{1:t+1}).
$$

Only the query for the new position must be fresh. The old keys and values are reused.

![Figure 2: What is cached in each layer](../zh/images/day17/kv-cache-per-layer.png)
*Caption: At decode time, each layer keeps a history of key and value tensors. For the next token, the layer computes only the new query, appends the new key and value, and attends over the stored history.*

A useful nuance is that the cache is stored **per layer** and usually **per attention head**. It is not one global memory for the whole model. Every Transformer layer has its own representation space, so each layer needs its own old keys and values.

This is why KV cache can consume so much memory. You are storing activations for every prior token, at every layer, for both keys and values.

A minimal decode sketch looks like this:

```python
for layer in transformer_layers:
    q_new = project_query(x_new)
    k_new = project_key(x_new)
    v_new = project_value(x_new)

    K_cache[layer].append(k_new)
    V_cache[layer].append(v_new)

    x_new = attention(q_new, K_cache[layer], V_cache[layer])
```

The pseudocode hides many details, but it captures the essential point: **append once, reuse many times.**

---

> **Why only cache K and V, not Q?**
>
> Look at the attention formula: softmax(QK^T / sqrt(d_k)) V
>
> - **Q** is the "question asker" -- only the **new** token needs to ask "which previous tokens should I attend to?" Old tokens' questions are irrelevant for the current step.
> - **K** is the "index tag" for every token -- the new token needs to compare against ALL previous tokens' keys. So we must keep all old K.
> - **V** is the "actual content" -- after attention weights are computed, we retrieve content from ALL previous tokens' values. So we must keep all old V.
>
> **Library analogy:** You are searching a library. Q = your current question (only need the latest one). K = every book's index tag (must keep all). V = every book's content (must keep all). You do not need to remember what questions you asked before -- but you always need access to the full catalog and book collection.


## 3. Prefill versus decode, two very different inference phases

Inference is easier to understand if we split it into two phases.

### 3.1 Prefill

The **prefill** phase processes the entire prompt. If the user sends a 4,000-token context, the model still has to run that context through all layers to build the initial cache. This phase is compute-heavy because there is lots of matrix multiplication and attention over many tokens at once.

### 3.2 Decode

The **decode** phase begins after the prompt is encoded. Now the model generates one new token at a time. This phase uses the existing cache and extends it incrementally. Compute per step is smaller, but memory traffic becomes dominant because every new query must read a large pile of cached keys and values.

> **Prefill vs Decode speed — which is faster?**
>
> Neither is "faster" in absolute terms -- they have different bottlenecks:
>
> - **Prefill** processes many tokens in parallel -- high throughput (tokens/sec), but heavy compute (lots of matrix multiplications)
> - **Decode** generates one token at a time -- low throughput (tokens/sec), bottlenecked by memory bandwidth (reading the growing KV cache each step)
>
> **Concrete example:** Prompt = 2,000 tokens, generating 200 tokens back
> - Prefill: process all 2,000 tokens at once, maybe 0.5 seconds
> - Decode: generate 200 tokens sequentially, maybe 2.0 seconds
> - Decode takes longer overall even though each step does less compute
>
> **One sentence:** Prefill is fast and bursty (parallel), Decode is slow and steady (sequential) -- long generation makes Decode the dominant bottleneck.

One important clarification: we normally talk about KV cache for **inference**, not for ordinary training. During training, the model processes full sequences in parallel and backpropagates through them, so storing a growing decode-style cache is not the central optimization.

This prefill/decode distinction matters a lot in serving. New requests often spend time in prefill, while long generations are dominated by decode. If you want to optimize latency, you need to know which phase is the bottleneck.

A simple analogy is cooking. Prefill is like chopping all the vegetables and preparing the sauce. Decode is like plating one dish after another using ingredients you already prepared. The second phase is faster per dish, but you still need space on the counter to keep everything organized.

---

## 4. Why KV cache speeds things up

![Figure: Why KV Cache Speeds Up Inference](../zh/images/day17/kv-cache-speedup.png)
*Caption: Without KV cache, every decoding step recomputes all previous tokens' projections. With KV cache, each token is computed only once — reducing 22 computations to just 4 in this example (~5.5x speedup).*

Without KV cache, the model recomputes old projections again and again. With KV cache, each token's key and value are computed once and then reused.

For a single new token at step $t$, the dominant attention score computation is between one query and $t$ cached keys, giving work proportional to $O(t)$ for that head, rather than recomputing all $t$ queries, all $t$ keys, and all $t$ values for the whole prefix again. Across the full decoding process, the savings are enormous in practice.

What the cache does **not** do is make attention free. The new query still needs to compare against all previous keys, so the context length still matters. KV cache turns “recompute everything” into “reuse history, scan history once per step.” That is a huge win, but not magic.

This is one reason long-context models remain expensive even with caching. If your context window jumps from 8K to 128K, the cache grows and each decode step has to read much more memory.

---

## 5. The hidden cost, memory

KV cache is a time-memory trade-off. We save compute by storing intermediate tensors, and the bill shows up in VRAM.

A rough memory estimate per sequence is

$$
\text{Cache bytes} \approx 2 \times L \times T \times H_{kv} \times D \times b,
$$

where:

- $L$ is the number of layers,
- $T$ is the number of cached tokens,
- $H_{kv}$ is the number of key-value heads,
- $D$ is head dimension,
- $b$ is bytes per element,
- and the factor 2 comes from storing both keys and values.

For fp16, $b = 2$. Even with grouped-query attention or multi-query attention reducing the number of KV heads, the memory still grows linearly with context length.

![Figure 3: Memory growth of the cache](../zh/images/day17/kv-cache-memory-growth.png)
*Caption: KV cache reduces recomputation but memory usage still scales linearly with context length. That becomes a major bottleneck for long prompts and large models.*

This trade-off creates several real-world problems:

- long contexts reduce the number of requests that fit in GPU memory,
- large batch sizes become harder,
- memory bandwidth can dominate latency,
- fragmentation becomes painful when many requests have different lengths.

> **What does "fragmentation" mean here?**
>
> When serving multiple users simultaneously, each request generates a KV cache of different size:
> - User A asks a short question → small cache (maybe 100 tokens)
> - User B sends a long document → huge cache (maybe 4000 tokens)
> - User C has a medium conversation → medium cache (maybe 500 tokens)
>
> If you pre-allocate a fixed-size memory block for each request, you get two problems:
> - **Over-allocation**: User A's block wastes most of its space
> - **External fragmentation**: freed blocks are different sizes, hard to reuse
>
> **Concrete example:** Imagine GPU memory as a parking lot. You reserve 4000 spaces for every car, but most cars only need 100. Soon the lot is "full" even though most spaces are empty. Paged attention solves this by using small fixed-size "pages" (like virtual memory in an OS) that can be flexibly assigned to any request.

So KV cache is not just an optimization trick. It reshapes the whole systems problem of LLM serving.

---

## 6. Why batching becomes complicated

Batching sounds easy in training because sequences are padded and processed together. In serving, requests arrive at different times and generate different numbers of tokens. One user may stop after 20 tokens, another may continue for 800 tokens. Their cache footprints evolve dynamically.

This is where naïve memory allocation breaks down. If every request gets one giant contiguous block for all possible future tokens, memory is wasted. If the blocks are resized often, fragmentation grows.

That is why high-performance engines introduced more sophisticated cache management. A good example is **paged attention**, popularized by vLLM. Instead of storing each sequence in one monolithic array, the system stores cache entries in smaller blocks that can be assigned and reused more flexibly.

![Figure 4: Paged attention organizes cache into blocks](../zh/images/day17/paged-attention-blocks.png)
*Caption: Paged attention stores KV cache in blocks rather than requiring one large contiguous memory region per request. This reduces fragmentation and improves utilization in multi-request serving.*

Think of it like virtual memory in an operating system. Rather than demanding one perfectly continuous warehouse shelf for every customer, you store items in labeled bins and keep a map. The customer sees one logical sequence, while the system gets much better packing efficiency.

---

## 7. Variants and related techniques

KV cache sits inside a broader family of inference optimizations.

### 7.1 Multi-query attention and grouped-query attention

In standard multi-head attention, every head has its own keys and values. **Multi-query attention (MQA)** shares keys and values across heads, while **grouped-query attention (GQA)** shares them within groups. The main motivation is to reduce cache size and memory bandwidth during decoding.

The trade-off is architectural. Sharing KV tensors can slightly reduce representational flexibility, but the serving gains are often worth it. This is one reason many modern LLMs adopt GQA.

### 7.2 Prefix caching

If many requests share the same prompt prefix, a system may reuse the prefill results rather than rebuilding the same cache repeatedly. This is especially useful for chat templates, long system prompts, or repeated RAG contexts.

### 7.3 Quantized KV cache

Just as model weights can be quantized, cached activations can sometimes be stored in lower precision. That saves memory, though it introduces engineering and accuracy trade-offs. Lower precision may be acceptable in some workloads and risky in others.

---

## 8. Common misconceptions

### Misconception 1: “KV cache makes decoding constant time.”

No. It removes redundant recomputation, but each new token still attends over the full cached history. Decode cost per step still grows with context length.

### Misconception 2: “The cache stores the model's knowledge.”

Not really. The model's learned knowledge lives in parameters. The KV cache stores **request-specific intermediate activations** for the current context.

### Misconception 3: “Cache only matters for long prompts.”

Long prompts amplify the effect, but even ordinary chat inference depends heavily on caching. Without it, token-by-token generation would feel dramatically slower.

### Misconception 4: “Cache is free if you have enough VRAM.”

Also no. Bandwidth matters as much as capacity. A system can have enough raw memory yet still become decode-bound because reading the cache every step is expensive.

---

## 9. Practical guidance

If you are building or evaluating an LLM system, here are the operational questions KV cache should trigger in your mind:

1. **How long are the prompts and generations?** Long contexts increase both cache size and decode cost.
2. **What attention variant does the model use?** GQA and MQA can materially reduce cache pressure.
3. **What is the serving pattern?** Many concurrent chats stress allocation and fragmentation.
4. **Is latency dominated by prefill or decode?** The answer changes the optimization strategy.
5. **Can shared prefixes be reused?** Prefix caching can be a large practical win.

For many real applications, the performance story of an LLM is not just “how many parameters?” but “how efficiently can the runtime manage KV cache?”

---

## 10. Further reading and reflection

### Further reading

- Tri Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
- vLLM paper and blog posts on paged attention
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (multi-query attention)

### Reflection question

If future architectures reduce or eliminate explicit attention-state caching, what trade-offs would they need to accept in expressiveness, training stability, or hardware efficiency?

---

## Closing takeaway

KV cache is one of those ideas that feels obvious after you learn it. Of course you should reuse past keys and values. But its importance is hard to overstate. It is the reason decoder-only Transformers can serve interactive text generation at all, and it is also the reason long-context inference becomes a memory systems problem.

So the compact summary is this: **KV cache turns repeated computation into stored state.** That makes generation fast enough to be useful, while shifting the bottleneck from arithmetic toward memory capacity, memory bandwidth, and runtime scheduling. If you understand that trade-off, you understand a surprisingly large fraction of modern LLM inference engineering.
