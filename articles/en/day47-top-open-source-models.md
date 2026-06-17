# Day 47: Top Open-Source Models In Depth — Learning Gemma 4, Qwen 3, and DeepSeek V4 Through Architecture

> **Core question**: Do not only ask "which open model is strongest?" A better question is: why do these models look the way they do, and where do they push the trade-offs among compute, memory, long context, multimodality, MoE routing, and edge deployment?

---

## Opening

In the previous article, we covered the open vs closed model landscape. Today we will not treat Gemma, Qwen, and DeepSeek as product names to memorize. We will treat them as three architectural paths:

- **Gemma 4**: edge-first design, prioritizing "can this run on ordinary devices?"
- **Qwen 3**: capability-matrix design, using both dense and MoE models to cover many scales and tasks, with unified thinking / non-thinking modes.
- **DeepSeek V4**: long-context and reasoning-efficiency design, using aggressive sparsity, compressed attention, and systems engineering to make 1M context routine.

The goal is more general than choosing among three model families. After reading this article, you should be able to inspect a new open model and infer what it is actually good for from its architecture table, rather than following leaderboard rankings blindly.

---

## 1. A Coordinate System for Reading Models

An LLM's architectural personality is rarely determined by one metric. It usually emerges from five groups of trade-offs:

| Dimension | Question to ask | What it affects |
|-----------|-----------------|-----------------|
| Parameter organization | Dense or MoE? How far apart are total and active parameters? | Knowledge capacity, inference cost, deployment barrier |
| Attention | MHA, GQA, MLA, sparse attention, compressed attention? | Long-context cost, KV cache, throughput |
| Long context | RoPE extrapolation, sliding window, compressed KV, or long-context training? | RAG, repository understanding, long-document tasks |
| Multimodality | Separate encoder, lightweight embedder, or encoder-free? | Vision/audio latency, fine-tuning complexity |
| Post-training | SFT, RLHF/RLAIF, thinking mode, agent data? | Instruction following, reasoning depth, tool use |

Many model writeups only say "the score is high." In engineering, the more important question is: **what cost was paid to get that score?**

For example:

- A 235B MoE model with 22B active parameters per token has compute closer to a 22B dense model, but memory still has to hold 235B parameters.
- A 1M-context model with full KV cache will quickly explode in memory and bandwidth; it needs compressed or sparse attention.
- An edge model cannot simply pile up parameters. It needs co-design across quantization, local attention, multimodal input paths, and inference runtime.

That is why we compare Gemma, Qwen, and DeepSeek by architecture rather than by slogans.

---

## 2. Gemma 4: Edge Models Are Not Just Smaller Models

### 2.1 Why: Why Make Gemma Edge-First?

Many open models assume you have one or more datacenter GPUs. Gemma starts from a different assumption: **the model should enter phones, laptops, browsers, edge devices, and local agents.**

That creates two hard constraints:

1. **Small memory budget**: phones and ordinary laptops cannot afford huge weights plus huge long-context KV cache.
2. **Latency matters more than peak capability**: edge apps often need immediate response and cannot send every request to the cloud.

So Gemma's central question is not "how large can the biggest model be?" It is "how much capability can we preserve under constrained hardware?"

### 2.2 How: From Gemma 3 to Gemma 4

Gemma 3 already showed this direction. It covered 1B, 4B, 12B, and 27B sizes, added vision understanding, supported at least 128K context, and reduced long-context KV-cache pressure by increasing the ratio of local to global attention layers and keeping local attention spans short.

Gemma 4 pushes further toward local agents. Official launch materials emphasize:

- **Apache 2.0**: friendly for commercial use.
- **Edge agents**: multi-step planning, offline code generation, visual/audio processing on device.
- **Encoder-free multimodality in Gemma 4 12B**: traditional multimodal models often use separate vision/audio encoders before projecting features into the LLM. Gemma 4 12B feeds multimodal inputs more directly into a decoder-only backbone, reducing latency and memory fragmentation.

### 2.3 What: The Key Architectural Ideas

![Figure 2: Three key architectural designs of Gemma 4 edge models](../zh/images/day47/gemma-architecture.png)
*Figure 2: Hybrid attention, GQA, and encoder-free multimodal input — the three pillars of edge-side model design.*

**1. Hybrid local + global attention**

Gemma 3 uses a pattern similar to five local sliding-window attention layers followed by one global attention layer. Local layers see nearby tokens; global layers periodically integrate full-context information.

Intuitively:

```text
Local layers:  cheaply process nearby context
Global layer:  occasionally synchronize global information
```

This is similar to how humans read long documents: most reasoning happens locally within paragraphs, while occasional passes integrate the whole structure. The model gives up the freedom of full attention in every layer, but gains lower KV cache cost and more stable long-context behavior.

**2. GQA reduces KV cache**

Grouped Query Attention lets multiple query-head groups share fewer key/value heads. This does not magically make the model smarter; it makes KV cache smaller and inference faster. For edge models, memory bandwidth often becomes the bottleneck before raw FLOPs do.

A common misunderstanding needs clarifying: **"sharing" does not mean sharing across different users or requests. It means that within a single token at a single layer, multiple query heads share the same K/V.**

In standard Multi-Head Attention (MHA), every query head has its own independent K head and V head. For example, 32 query heads require 32 K heads + 32 V heads, and the KV cache stores 32 copies.

GQA groups the 32 query heads into several groups (e.g., 8 groups), where each group of 4 query heads **shares 1 K head and 1 V head**. The result: only 8 K/V heads, and the KV cache stores only 8 copies — a 75% memory saving.

The critical point: **all query heads still independently compute attention.** Each head uses its own distinct Q matrix to dot-product with the shared K, producing different attention patterns. What is shared is the K/V storage, not the attention computation result.

An analogy: MHA is 32 people each carrying their own reference book; GQA is 32 people split into 8 groups of 4, each group sharing one book — but each person **reads it differently** (different Q), so they reach different conclusions.

This works because different attention heads learn K/V representations with significant redundancy — many heads' K/V are nearly identical. GQA explicitly merges this redundancy, with almost no quality loss but a dramatic reduction in inference-time memory and bandwidth pressure.

**3. A shorter multimodal input path**

Gemma 4 12B's encoder-free direction is important. To understand why this saves so much, first look at the traditional VLM input path:

```text
Traditional VLM: Image → Vision Encoder (ViT/SigLIP, hundreds of millions to 1B params) → Projector (MLP/Q-Former) → LLM token space
```

The Vision Encoder is a complete independent Transformer designed to "see" images. It converts pixels into feature vectors, and then the Projector aligns those vectors into the LLM's embedding space. Two large modules, a long path.

Gemma 4 12B takes a different approach: instead of a full ViT, it uses a **lightweight patch embedding layer** — the image is sliced into patches, each patch is mapped to a vector by a shallow CNN or linear layer, and the result is fed directly into the LLM. No hundreds-of-millions-parameter vision transformer, no separate projector alignment step. Audio works similarly: instead of audio → Mel spectrogram → independent audio encoder → projector → LLM, Gemma 4 splits the audio waveform into short frames and maps them to tokens via a lightweight projection layer.

**Why does this work?** The key insight is that the LLM itself has enough capacity to process these inputs, provided they are aligned to the token space. Traditional VLMs used heavy encoders because early LLMs were not strong enough — they needed the encoder to first "refine" visual/audio features. But when you have a 12B-class LLM, it can learn to handle these shallow embeddings on its own.

**But this means training must be end-to-end.** During training, images are sliced into patches, each patch passes through the lightweight embedder to become a vector, and these vectors are concatenated with text token embeddings in the same sequence. The model is trained with standard next-token prediction — it must learn to understand these patch vectors, or the loss will not go down. After enough image-text paired training, the LLM's attention layers naturally learn "this patch vector represents a red pixel block" or "this combination of patches is a cat."

This is fundamentally different from traditional VLMs: there, the Vision Encoder is pre-trained and already knows how to convert pixels into meaningful features. The Projector then "translates" these features for the LLM. The LLM itself never sees raw pixels — it only sees pre-translated high-level features. The cost of going encoder-free is that you need massive amounts of image-text and audio-text paired training data, because the LLM must learn visual understanding from scratch rather than standing on the shoulders of a ViT. But the benefit is that once learned, you no longer need that independent encoder at inference time.

**Why does this also reduce memory fragmentation on edge devices?** In the traditional multimodal path, the device must simultaneously run several large modules with very different memory characteristics:

```text
Image → Vision Encoder → Projector → LLM
Audio → Audio Encoder → Projector → LLM
```

This causes three types of fragmentation:

1. **Many modules with different lifetimes** — Vision encoder, audio encoder, projector, and LLM each have their own weights, activations, and temporary buffers. Some are only used during pre-processing, others throughout generation. The edge memory allocator must constantly allocate and free blocks of different sizes, leaving many non-contiguous gaps over time.
2. **Large buffer size variance** — Vision encoder produces patch features, attention maps, and hidden states; audio encoder produces spectral or frame-level features; LLM produces token embeddings and KV cache. These tensors have different sizes, shapes, and alignment requirements, making it hard for the memory planner to reuse buffers ahead of time.
3. **Hard to reuse across runtimes** — If vision encoder, audio encoder, and LLM use different operator paths, the memory planner cannot easily coordinate buffer reuse. On edge devices with mixed GPU/NPU/CPU execution, this also involves cross-device copies and temporary staging buffers.

The lightweight embedder + wave projection approach unifies inputs earlier into `image patches / audio frames → LLM token embeddings → LLM trunk`, so the front end no longer maintains full intermediate activations from a large independent encoder — everything enters the same LLM memory pattern. Memory allocation becomes more concentrated and predictable, and buffer reuse is easier. **It is not that projection itself magically "defrags memory" — rather, it reduces the number of independent large modules and heterogeneous intermediate tensors, making inference look more like a single LLM pipeline.**

Beyond latency and memory, there is a third benefit: during LoRA or full fine-tuning, you do not need to coordinate multiple large frozen encoders.

### 2.4 What Do We Learn?

Gemma teaches a key lesson: **a small model is not just a shrunken large model. A good edge model is co-designed across attention, KV cache, multimodal inputs, quantization, and runtime.**

When a model claims to be edge-friendly, do not only look at parameter count. Ask:

- How does it control KV cache?
- Is attention global, local, or hybrid?
- Does multimodality depend on heavy encoders?
- Are real edge paths provided, such as LiteRT, MediaPipe, llama.cpp, or Ollama?

> **What are LiteRT and MediaPipe?**
>
> **LiteRT** (formerly TensorFlow Lite / TFLite, renamed in 2024) is Google's on-device inference runtime, designed for phones, tablets, and IoT devices. It converts models to .tflite format and executes inference via CPU/GPU/NPU Delegates, with INT8/INT4 quantization support. Gemma 4 E2B/E4B officially provide LiteRT deployment paths for NPU-accelerated inference on phones.
>
> **MediaPipe** is Google's cross-platform ML pipeline framework — not just an inference engine, but a tool for "stringing models into applications." For example, if you want to build a "photo → face detection → expression recognition → output" pipeline, MediaPipe helps you chain multiple models together, handling camera input, preprocessing, and postprocessing. In 2024 it integrated LLM inference capabilities (MediaPipe LLM Inference API), supporting Gemma and other small models on phones.
>
> Simple analogy: **LiteRT ≈ on-device vLLM/SGLang** (efficiently runs models), **MediaPipe ≈ on-device LangChain** (chains models with I/O, preprocessing, and postprocessing into complete applications). MediaPipe can use LiteRT as its inference backend.
>
> These are mentioned because Gemma 4 is not just "small" — it comes with a complete edge deployment toolchain. A model, no matter how small, cannot run well without good runtime and pipeline tooling.

---

## 3. Qwen 3: The Point Is Not "Many Models," but a Unified Dense/MoE Design

### 3.1 Why: Why Does Qwen Need So Many Models?

Qwen follows a different path from Gemma. Gemma asks how to run on devices. Qwen asks how to cover as many capability boundaries as possible: general reasoning, coding, multilinguality, agents, long context, and different memory tiers.

That is why Qwen 3 includes both dense and MoE models:

- **Dense models**: simpler structure, easier deployment and fine-tuning, good for small and mid-sized deterministic deployments. "Simpler" here is relative to MoE — every token passes through all parameters, with no router, no expert selection, just linearly running through each layer. This means: no need for router kernels or expert-parallelism during deployment; gradients flow directly to all weights during fine-tuning without auxiliary losses to constrain routing; VRAM and FLOPs are fixed and predictable; and frameworks like vLLM, llama.cpp, and Ollama have more mature support for dense models. Fewer complexity dimensions means fewer bugs and faster iteration.
- **MoE models**: larger total parameter capacity, but only part of the experts activate per token, improving capability per unit of compute.

### 3.2 How: Qwen 3's Base Architecture

![Figure 3: Qwen 3 Transformer Block — unified Dense and MoE design](../zh/images/day47/qwen-architecture.png)
*Figure 3: The same transformer structure, where the FFN can be either Dense (all params computed) or MoE (router selects top-8 experts). This is the core of Qwen 3's "one architecture, two model types" design.*

The Qwen 3 technical report describes a familiar modern decoder-only architecture: GQA, SwiGLU, RoPE, RMSNorm, and pre-normalization. If these terms are new to you, here is a quick walkthrough:

> **RoPE (Rotary Positional Embedding)**
>
> Transformers do not inherently know token order — you need some way to tell the model "this word is at position 5, that one at position 100." The old approach was to add a fixed or learnable position vector to each token (Positional Encoding). RoPE takes a different approach: it applies rotation matrices to the Q and K vectors, encoding position into the angles of the vectors.
>
> Why is this good? The relative positional relationship between two tokens ("how far apart they are") is directly reflected in the Q·K dot product result, without needing extra position parameters. This makes RoPE especially suited for long contexts — different rotation frequencies capture positional relationships at different distances, and it naturally supports context length extrapolation.
>
> **RMSNorm (Root Mean Square Normalization)**
>
> Traditional LayerNorm does two things: subtract the mean and divide by standard deviation (normalization), then apply two learnable parameters for scaling and shifting. RMSNorm drops the "subtract mean" and "shift" steps, only scaling by the RMS (root mean square).
>
> Why? At large training scale, the mean-subtraction step has non-trivial computational cost, yet contributes little to final quality. RMSNorm needs only one multiplication and one square root, making it ~10-50% faster than LayerNorm with nearly identical results. Nearly every open-source LLM since 2024 (Llama, Qwen, Gemma, DeepSeek) uses RMSNorm instead of LayerNorm.
>
> **Pre-normalization**
>
> This tells you where RMSNorm goes in the transformer layer. Pre-norm means: normalize first, then enter attention or FFN.
>
> ```text
> Pre-norm:  x → RMSNorm → Attention → + x (residual)
> Post-norm: x → Attention → + x (residual) → RMSNorm
> ```
>
> Pre-norm is more stable during training — gradients can flow directly back to early layers through residual connections, without being distorted or vanishing through too many normalization + non-linear transforms. Post-norm frequently caused training collapse in early deep models. Modern LLMs almost universally use pre-norm.
>
> **SwiGLU (Swish-Gated Linear Unit)**
>
> This is the activation function design for the FFN (Feed-Forward Network) layer. A standard FFN is: `x → Linear → ReLU → Linear`. SwiGLU changes it to: `x → two parallel Linears → apply Swish to one → multiply the two → Linear`.
>
> Intuitively: SwiGLU = gating + non-linearity. One branch computes the "value," the other computes "how wide the gate opens," and their product determines the output. The Swish function (`x · sigmoid(βx)`) is smoother than ReLU — it does not have the "cliff" near zero that ReLU has, making gradients friendlier.
>
> The cost is one extra Linear layer (because you need two parallel branches), but experiments show SwiGLU outperforms ReLU and GeLU variants under equal parameter budgets. Llama, Qwen, and Gemma all use SwiGLU.

We already covered GQA earlier. Two details truly matter here:

1. **QKV bias is removed, QK-Norm is introduced**  
   QK-Norm stabilizes the scale of attention queries and keys, reducing the risk of exploding attention logits during large-scale training.

2. **MoE and dense models share the same foundation**  
   Qwen 3 MoE is not a separate architecture. It uses the same transformer backbone and replaces dense FFN blocks with multiple expert FFNs, then lets a router choose experts for each token.

### 3.3 What: What Is Qwen 3 MoE Actually Doing?

Take Qwen3-235B-A22B:

| Metric | Value |
|--------|-------|
| Total parameters | 235B |
| Active parameters per token | 22B |
| Layers | 94 |
| Attention heads | Q 64 / KV 4 |
| Experts | 128 total / 8 activated |
| Context | 128K |

Three points matter.

**1. MoE saves compute, not weight storage**

235B-A22B means the model has 235B parameters in total, but each token activates about 22B. Matrix multiplication cost is closer to a 22B dense model, but memory still needs to hold 235B of weights.

So MoE is excellent for cloud and multi-GPU serving. It does not automatically make a model easy to run on a single local GPU.

**2. 128 experts, 8 selected per token**

The router selects top-8 experts per token. Ideally, experts specialize into different capability regions: code, math, multilingual text, formatting, tool use, domain knowledge, and so on. But routing creates two problems:

- if a few experts are selected too often, load becomes imbalanced;
- if we force load balancing too aggressively, model quality can suffer.

Qwen 3 uses global-batch load balancing loss to encourage expert specialization. This is not a free trick; it is a compromise between specialization and hardware load balance.

**3. No shared experts**

The Qwen 3 technical report notes that Qwen3-MoE removes the shared experts used in Qwen2.5-MoE. Shared experts usually provide a general fallback path. Removing them makes the model rely more strongly on routing token to the right experts. That can create cleaner specialization, but makes router training more important.

### 3.4 Thinking / Non-Thinking: Inference Budget Control

One important productized design in Qwen 3 is unified thinking and non-thinking modes. The underlying question is: **when should the same model answer quickly, and when should it spend more tokens reasoning?**

This is not just a prompt trick. It is a way to control test-time compute:

```text
Simple question: fewer reasoning tokens -> lower latency, lower cost
Hard question:   more reasoning tokens -> higher accuracy, higher cost
```

For developers, this is important because model selection is no longer only "which model?" It is also "how much reasoning budget should we allocate?"

### 3.5 What Do We Learn?

Qwen teaches that **MoE is not about making parameter count as large as possible. It is about the joint design of routing, expert granularity, load balancing, context length, and post-training.**

When reading an MoE model card, ask:

- What are the total and active parameters?
- How many experts exist per layer, and how many are activated per token?
- Are there shared experts?
- Is load balancing done with auxiliary loss, global-batch loss, or another mechanism?
- Is thinking mode only a prompt convention, or is it supported by training and serving?

---

## 4. DeepSeek V4: Making 1M Context Cheap Is the Real Architecture Problem

### 4.1 Why: Why Did Long Context Become DeepSeek's Main Battlefield?

For ordinary chat, 8K or 32K context is often enough. But agents, repository-level coding, enterprise knowledge bases, long-document analysis, and long-horizon tasks need much longer context. The issue is that transformer attention and KV cache are extremely sensitive to sequence length.

When context grows from 128K to 1M, the hard part is not merely fitting tokens into the window. The hard parts are:

- prefill becomes expensive;
- every decode step has to access a huge historical KV cache;
- KV cache consumes large amounts of memory;
- multi-user serving turns cache scheduling and prefix reuse into system bottlenecks.

DeepSeek V4 focuses on making **1M context not a demo feature, but a routinely serviceable capability.**

### 4.2 How: From DeepSeek V3 to V4

DeepSeek V3 established three foundations:

- **DeepSeekMoE**: 671B total / 37B active, using sparse experts to reduce per-token compute.
- **MLA (Multi-head Latent Attention)**: compresses key/value representation to reduce KV cache.
- **MTP (Multi-Token Prediction)**: predicts multiple future tokens during training, strengthens the learning signal, and can support speculative decoding.

DeepSeek V4 builds on that foundation. Its technical report introduces two models:

| Model | Total parameters | Active parameters | Context |
|-------|------------------|-------------------|---------|
| DeepSeek-V4-Pro | 1.6T | 49B | 1M |
| DeepSeek-V4-Flash | 284B | 13B | 1M |

The major upgrades include:

- **hybrid CSA + HCA attention**;
- **mHC (Manifold-Constrained Hyper-Connections)**;
- **Muon optimizer**;
- lower precision for routed experts;
- more sophisticated KV cache management and long-context serving systems.

### 4.3 What: Why CSA + HCA Matters

The DeepSeek V4 report describes two attention mechanisms:

**CSA (Compressed Sparse Attention)**  
First compress the KV cache of every m tokens into one entry, then let each query attend only to top-k compressed entries.

```text
long sequence -> compressed KV blocks -> sparse selection of relevant blocks -> attention
```

It does two things at once:

- compresses sequence length;
- avoids dense attention over every historical block.

**HCA (Heavily Compressed Attention)**  
Compresses much longer spans into a single KV entry, but keeps dense attention over those compressed entries. It provides a coarser global memory.

The combination can be understood as:

```text
CSA: keep finer-grained retrievable history, but look only at relevant parts
HCA: keep coarser global summaries of very long history
```

This is stronger than simple sliding-window attention because a pure sliding window discards far-away details. It is cheaper than full global attention because full attention is too expensive at 1M context.

### 4.4 mHC: Why Upgrade Residual Connections?

Transformer residual connections look ordinary, but when models become extremely deep, sparse, and large-scale, stable signal propagation across layers matters.

DeepSeek V4 uses mHC to strengthen conventional residual connections. Roughly, instead of simply adding the previous layer output back, it mixes multiple state paths with constrained mappings. The report says the residual mapping is constrained to a manifold of doubly stochastic matrices, improving stability across layers.

This reminds us that late-stage scaling innovations do not only happen in attention and MoE. They also happen in apparently small details such as connection structures and optimizers.

### 4.5 Muon Optimizer: Training Efficiency Is Part of the Architecture

DeepSeek V4 uses the Muon optimizer for most modules, while keeping AdamW for embeddings, prediction heads, RMSNorm, and several special parameters. The goal is faster convergence and better training stability.

You do not need to memorize Muon's algorithm to get the lesson: **at trillion-scale MoE, optimizer choice, parallelism, checkpointing, kernels, and KV-cache storage are no longer mere engineering details. They determine whether the model can exist as a deployable system.**

### 4.6 What Do We Learn?

DeepSeek teaches that **long context is not a large number in the context-window field. It is a joint solution across attention, KV cache, sparse routing, low precision, and inference systems.**

When a model advertises 1M context, ask:

- Was it actually trained for long context?
- Is attention dense, sparse, or compressed sparse?
- How is KV cache compressed and stored?
- What is the single-token decode cost at 1M?
- Are long-context benchmarks synthetic retrieval tasks, or real document/code/agent tasks?

---

## 5. Putting the Three Paths Together

![Figure 1: Gemma 4 vs Qwen 3 vs DeepSeek V4 capability profiles](../zh/images/day47/triple-comparison.png)
*Figure 1: Capability profiles of the three open-source families. The important part is not which area is larger, but which architectural trade-offs create each shape.*

| Question | Gemma 4 | Qwen 3 | DeepSeek V4 |
|----------|---------|--------|-------------|
| Primary goal | Edge usability, local agents | Broad capability matrix | 1M context and reasoning efficiency |
| Parameter path | Small/mid models + edge optimization | Dense + MoE matrix | Large MoE + high sparsity |
| Attention focus | Local/global hybrid, GQA | GQA, QK-Norm, RoPE | CSA + HCA, compressed/sparse attention |
| MoE focus | Lightweight MoE in some variants | 128 experts / top-8, global-batch balance | DeepSeekMoE, high total/active ratio |
| Multimodal path | Edge vision/audio, 12B encoder-free | Vision in selected models, strong code/agent focus | Mostly text, reasoning, long context |
| Best-fit scenarios | Phones, laptops, offline use, local privacy | Multilingual, coding, general agents | Long documents, repositories, low-cost reasoning, complex reasoning |
| Main risk | Lower absolute ceiling | Variant complexity, deployment barrier | System complexity, latency and serving burden |

These are not simply stronger or weaker versions of one another. They answer different questions:

- Gemma 4 asks: **can we bring agents onto the device?**
- Qwen 3 asks: **can one model matrix cover the widest range of capability scenarios?**
- DeepSeek V4 asks: **can ultra-long context and strong reasoning become cheap enough to use routinely?**

---

## 6. Why Do Some Models Support Multimodality While Others Do Not?

Multimodality is not as simple as attaching a camera to a text model. At minimum, it has to solve three problems:

1. **Input representation**: image patches, audio frames, and video frames must become token-like representations the LLM can process.
2. **Cross-modal alignment**: the model must learn how visual/audio features correspond to language concepts, such as how a table in an image maps to rows and columns in text.
3. **Inference cost**: images and audio introduce extra tokens or features, making long context, KV cache, and batching more expensive.

So whether a model family supports multimodality usually depends on its primary objective.

**Gemma 4 has strong reasons to put multimodality on the core path**, because its goal is edge agents. Phones, browsers, and local assistants naturally need cameras, screenshots, audio, and screen understanding. If Gemma were text-only, it would be much less useful as an on-device agent. That is why it emphasizes lightweight vision/audio input paths and explores encoder-free or near-encoder-free designs that reduce latency and memory pressure on device.

**Qwen 3 is "multimodal in selected variants, specialized in others."** Qwen's core idea is a model matrix rather than a single universal model. Some variants target vision-language tasks; others prioritize coding, agentic coding, math, multilinguality, or thinking modes. Vision requires extra data, encoders or embedders, alignment training, and evaluation. For a coding-agent model, spending training budget on repository data, tool-use traces, and code execution feedback can be more valuable than adding image input.

**DeepSeek V4 is mostly focused on text, reasoning, and long context**, not because multimodality is unimportant, but because its main architecture problem is already heavy: 1M context, compressed/sparse attention, MoE routing, KV cache, and low-cost serving. Multimodal inputs would add more tokens and cache pressure, competing directly with the goal of making ultra-long text context cheap. For DeepSeek's path, solving text long context and reasoning cost first is more coherent than adding vision/audio at the same time.

In short:

| Model family | Why is multimodality designed this way? |
|--------------|------------------------------------------|
| Gemma 4 | Edge agents need vision, audio, and screen understanding, so multimodality is part of the core path |
| Qwen 3 | The model matrix serves different tasks; vision is one branch, not a requirement for every variant |
| DeepSeek V4 | The technical budget goes into text reasoning, 1M context, and low-cost serving; multimodality would add KV/cache and training complexity |

The lesson is important: **lack of multimodality is not always a sign that a model is behind. Sometimes it means the architecture budget was deliberately spent on a different capability curve.**

---

## 7. Practical Selection: Start from Constraints, Not Leaderboards

### 7.1 If You Are Building a Local App

Start with Gemma.

Not because it is always strongest, but because its system assumptions match yours: edge runtime, quantization, low latency, multimodal input, and local API serving matter more than raw leaderboard scores.

Good fits:

- local writing assistants;
- offline meeting notes;
- mobile visual Q&A;
- privacy-sensitive personal knowledge bases;
- local coding or data-analysis tools.

### 7.2 If You Are Building a General AI Product

Start with Qwen.

Its strength is the completeness of the model matrix: small dense models, large MoE models, general models, coder models, non-thinking and thinking modes. You can start with smaller prototypes and move up the matrix without locking yourself into one isolated model.

Good fits:

- multilingual customer support;
- coding assistants;
- enterprise internal agents;
- industry models that need fine-tuning;
- products that need controlled cost today and an upgrade path later.

### 7.3 If You Need Long Context and Heavy Reasoning

Start with DeepSeek.

Especially for:

- repository-level understanding;
- large PDF / contract / research-report analysis;
- long-horizon agent tasks;
- cost-sensitive API workloads;
- math, science, and engineering reasoning.

DeepSeek's core value is not simply being cheap. Its cost advantage is architectural: MoE sparsity, KV compression, long-context attention, low-precision training/inference, and serving systems work together.

---

## 8. Common Misconceptions

### Misconception 1: "MoE is always cheaper"

MoE saves per-token compute, not necessarily memory. Large total parameters still require large weight storage. Expert parallelism can also introduce communication overhead.

### Misconception 2: "1M context means the model can read 1M well"

Not necessarily. Long-context ability has at least three layers:

- training and positional encoding support;
- attention and KV cache efficiency;
- real ability to retrieve, summarize, and reason over long documents.

Many models can fit long input, but cannot use long input reliably.

### Misconception 3: "Edge models are just weak models"

Edge models optimize a different objective: privacy, low latency, offline use, low cost, and embeddability. For many real products, those constraints matter more than a 2-3 point benchmark difference.

### Misconception 4: "Open models only copy closed models"

That is no longer true. Gemma pushes local agents, Qwen pushes open MoE matrices and agentic coding, and DeepSeek pushes long-context efficiency and sparse systems engineering. Open models are not merely cheaper substitutes; in several engineering directions, they are driving the frontier.

---

## 9. Further Reading

### Official Materials and Technical Reports

1. [Gemma 4 launch: Bring state-of-the-art agentic skills to the edge](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/)
2. [Gemma 4 12B: The Developer Guide](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/)
3. [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
4. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
5. [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
6. [Qwen3-Coder: Agentic Coding in the World](https://qwenlm.github.io/blog/qwen3-coder/)
7. [DeepSeek V4 Preview Release](https://api-docs.deepseek.com/news/news260424)
8. [DeepSeek-V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
9. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

---

## Reflection Questions

1. If you are choosing a model for a local knowledge base, why might maximum parameter count be the wrong first metric?
2. Qwen3-235B-A22B activates only 22B parameters per token. Why is it still not a good fit for an ordinary single consumer GPU?
3. DeepSeek V4's CSA + HCA and Gemma's local/global attention both reduce long-context cost. What does each approach sacrifice, and what does it preserve?
4. If future models all support thinking mode, should products price by input/output tokens, or by reasoning effort?

---

## Summary

| Model family | One-line positioning | The real architectural lesson |
|--------------|----------------------|-------------------------------|
| Gemma 4 | Edge-agent path | Edge capability comes from joint design across KV cache, local attention, multimodal input paths, and runtime |
| Qwen 3 | Open capability-matrix path | MoE is about total/active params, routing, expert granularity, load balancing, and reasoning budget |
| DeepSeek V4 | Long-context efficiency path | 1M context requires compressed/sparse attention, KV-cache engineering, low precision, and training-system co-design |

**Key takeaway**: The difference among top open models is no longer only "which benchmark is higher." Gemma, Qwen, and DeepSeek represent three paths: edge deployment, capability matrices, and long-context efficiency. The real learning value is understanding how LLM capability is always negotiated among compute, memory, latency, data, training stability, and serving systems.

---

*Day 47 of 60 | LLM Fundamentals*
*Word count: ~3300 | Reading time: ~18 minutes*
