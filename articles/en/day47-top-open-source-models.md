# Day 47: Top Open-Source Models In Depth — Gemma 4, Qwen 3, DeepSeek V4

> **Core Question**: The open-source model landscape is blooming, but you can't try them all. Which ones deserve your attention, and what is each one best at?

---

## Opening

In the previous article, we covered the overall open vs closed landscape. Today we zoom in on three of the most important open-source model families: **Gemma 4, Qwen 3, and DeepSeek V4**.

Why these three? Because they represent three distinct philosophies of open-source LLMs in 2026:

- **Gemma 4** (Google) — Comprehensive coverage from phones to workstations, most business-friendly license
- **Qwen 3** (Alibaba) — Strongest overall capability, the all-rounder for multilingual and coding tasks
- **DeepSeek V4** (DeepSeek AI) — Top-tier reasoning, the price disruptor

Choosing a model is like choosing a tool — there's no "best hammer," only "the right hammer for your specific nail." Understanding each model's design philosophy and strength boundaries matters more than memorizing benchmark scores.

---

## 1. Gemma 4 — Google's Open Hand

### 1.1 Design Philosophy: Full Coverage from Edge to Cloud

Gemma 4 was released on March 31, 2026, under the Apache 2.0 license (unrestricted commercial use). Its core design goal: **make frontier AI capability available on every device**.

What makes Gemma 4 unique is that it's not a single model but a product line covering different hardware tiers:

| Model | Parameters | Active Params | Context Window | Target Hardware |
|-------|-----------|---------------|----------------|-----------------|
| Gemma 4 E2B | 2B | 2B | 128K | Phones, IoT |
| Gemma 4 E4B | 4B | 4B | 128K | Edge devices, Raspberry Pi |
| Gemma 4 26B MoE | 26B | 3.8B | 256K | Consumer GPUs |
| Gemma 4 31B Dense | 31B | 31B | 256K | Workstations, servers |

#### Why This Matters

Most open-source models only offer "large" versions — you need an H100 to run them. Gemma 4 E2B can do on-device inference on phones, enabling a whole new category of applications: **offline AI assistants, privacy-first local document analysis, intelligent tools for no-network environments**.

The 26B MoE version is particularly clever: 26B total parameters but only 3.8B active per inference, so speed approximates a 4B model while capability approaches the 30B class. A consumer GPU (like RTX 4090) can run it smoothly.

### 1.2 Performance

Gemma 4 31B Dense on key benchmarks:

| Benchmark | Gemma 4 31B | Comparison |
|-----------|------------|------------|
| MMLU Pro | **85.2%** | Close to GPT-5.4 Mini |
| AIME 2026 | **89.2%** | Excellent math reasoning |
| LiveCodeBench v6 | **~80%** | Solid coding ability |
| Arena AI | #3 | Highest among open-source |

### 1.3 Strengths & Weaknesses

**Strengths:**
- **Apache 2.0 license** — most permissive commercial license, no reporting or restrictions
- **End-to-end coverage** — from 2B to 31B, one model family for all deployment scenarios
- **Great local deployment ecosystem** — Ollama, LM Studio, MediaPipe, LiteRT all supported
- **Multimodal** — supports text and image input; smaller models also support audio

**Weaknesses:**
- 31B Dense absolute capability still below Qwen 3.7 Max and DeepSeek V4 Pro
- Agent tool-use capability less specialized than Qwen 3-Coder
- Community fine-tune ecosystem less rich than Llama's

### 1.4 When to Choose Gemma 4?

- You need to run models **on phones/edge devices**
- You need **Apache 2.0 commercial certainty**
- You want **one model family from small to large**, unified tech stack

---

## 2. Qwen 3 — Alibaba's All-Arounder

### 2.1 Design Philosophy: Push Frontier Capability into Open Source

The Qwen (Tongyi Qianwen) series is one of the strongest open-source model families globally in 2026. Unlike Gemma's "cover all scenarios" strategy, Qwen's philosophy is: **be the best open-source model in every lane**.

Qwen 3's model matrix is extremely rich:

| Model | Type | Parameters | Context Window | Positioning |
|-------|------|-----------|----------------|-------------|
| Qwen 3.7 Max | MoE | Undisclosed | 1M | Strongest overall |
| Qwen 3.7 Plus | MoE | Undisclosed | 1M | High value + vision |
| Qwen 3-Coder 480B | MoE | 480B (A35B) | 256K | Coding-specialized |
| Qwen 3 235B | MoE | 235B (A22B) | 128K | General flagship |
| Qwen 3.5-397B | Dense | 397B | 128K | Research/deployment |
| Qwen 3-30B-A3B | MoE | 30B (A3B) | 128K | Consumer GPU |

### 2.2 Capability Highlights

**Reasoning:** Qwen 3.7 Max scores 92.4% on GPQA Diamond, beating Claude Opus 4.6. On HMMT 2026 (math competition), it scores 97.1, surpassing DeepSeek V4 Pro.

**Coding:** Qwen 3-Coder-480B is specifically trained for agentic coding, reaching Claude Sonnet-level performance on code generation and debugging. If you're building an AI coding assistant, this is the strongest open-source option.

**Multilingual:** Qwen 3.5 covers 201 languages and dialects, far more than any other open-source model. If your application targets non-English markets, Qwen is virtually the only choice.

### 2.3 Qwen's MoE Architecture In Depth

Qwen 3 heavily uses Mixture-of-Experts (MoE) architecture. Taking Qwen 3 235B as an example:

- **Total parameters**: 235B
- **Active parameters per inference**: 22B (~1/10)
- **Number of experts**: 128
- **Experts selected per token**: 8 (Top-8 routing)

This means while the model has 235B of knowledge capacity, each inference only computes 22B worth of work — **inference speed equals a 22B dense model, but capability far exceeds it**.

The trade-off is VRAM: you need to hold all 235B parameters (~470GB FP16, or ~120GB in 4-bit quantization), but computation only requires 22B of processing power.

### 2.4 Strengths & Weaknesses

**Strengths:**
- **Strongest overall capability** — top 3 open-source on virtually every benchmark
- **Richest model variants** — from 3B to 480B, from general-purpose to coding-specific
- **Apache 2.0 license** — commercially friendly
- **1M context window** — matches closed-source flagships
- **Unmatched multilingual coverage** — 201 languages

**Weaknesses:**
- Flagship models have high deployment barriers (235B needs ~4-8 A100s)
- Too many model variants creates selection complexity
- Documentation and developer experience less clear than Gemma and DeepSeek

### 2.5 When to Choose Qwen 3?

- You need **the strongest open-source general reasoning**
- Your application targets **multilingual markets**
- You're building an **AI coding assistant** (choose Qwen 3-Coder)
- You have GPU budget for large models

---

## 3. DeepSeek V4 — The Reasoning King & Price Disruptor

### 3.1 Design Philosophy: Outperform Through Engineering Efficiency

DeepSeek AI is a Chinese AI lab focused on open-source, renowned for "doing more with less compute." DeepSeek V4, released in April 2026, once again redefined the price-performance frontier.

| Model | Parameters | Active Params | Context Window | Type |
|-------|-----------|---------------|----------------|------|
| DeepSeek V4 Pro | 1.6T | 49B | 1M | General MoE |
| DeepSeek V4 Flash | Undisclosed | Undisclosed | 1M | Lightweight fast |
| DeepSeek R1 | 671B | 37B | 128K | Reasoning-specialized |

### 3.2 Capability Highlights

**Coding:** V4 Pro scores 80.6% on SWE-bench Verified, nearly matching Claude Opus 4.6 (80.8%). Considering it's 35-100x cheaper, this result is staggering.

**Math Reasoning:** DeepSeek R1 scores 99.4% on AIME 2026, among the highest of any model including closed-source. It proves: **frontier reasoning capability doesn't necessarily require trillion parameters and astronomical compute**.

**Pricing:** V4 Flash at $0.14/$0.28 per MTok — roughly 100x cheaper than GPT-5.5. If API cost is your product's core constraint, DeepSeek is almost a no-brainer.

### 3.3 DeepSeek's Engineering Innovations

DeepSeek's low prices don't come from cutting corners but from extreme engineering optimization:

**1. Aggressive MoE Design**
V4 Pro's 1.6T parameters with only 49B active (~3%) — this sparsity ratio is far higher than Qwen 3 235B's 9.4%. More aggressive sparsity means lower inference cost, but also makes training harder (routing instability, load balancing difficulty).

**2. Multi-Token Prediction (MTP)**
DeepSeek V4 uses MTP in both training and inference — the model predicts multiple tokens at once rather than one at a time. This dramatically increases inference throughput.

**3. FP8 Training**
DeepSeek was one of the first teams to use FP8 precision training at scale. Lower precision = less VRAM = lower training cost.

### 3.4 Strengths & Weaknesses

**Strengths:**
- **Top-tier reasoning** — competition-grade math problem capability
- **Lowest API pricing** — price-performance crushes all competitors
- **MIT license** — most permissive open-source license
- **OpenAI API-compatible** — zero migration cost
- **Extremely detailed technical reports** — invaluable for researchers

**Weaknesses:**
- **No multimodal** — text-only, cannot process images
- **No native embedding model** — RAG scenarios need a companion model
- **1.6T parameter deployment barrier** — V4 Pro self-deployment needs ~16 H100s
- **Higher latency** — real-time scenarios (chat, streaming) are slower than Gemini Flash

### 3.5 When to Choose DeepSeek?

- Your application is **extremely cost-sensitive**
- You need **top-tier math/logic reasoning** (choose R1)
- Your application is **text-only** (classification, summarization, code generation, Q&A)
- You do **AI research** and need detailed technical reports and reproducible methods

---

## 4. Head-to-Head: How to Choose Among the Three?

![Figure 1: Gemma 4 vs Qwen 3 vs DeepSeek V4 capability profiles](../zh/images/day47/triple-comparison.png)
*Figure 1: Capability profiles of the three open-source families across dimensions. Shape matters more than area — each family has a unique "fingerprint."*

| Dimension | Gemma 4 | Qwen 3 | DeepSeek V4 |
|-----------|---------|--------|-------------|
| General Reasoning | ★★★★☆ | ★★★★★ | ★★★★★ |
| Coding Ability | ★★★★☆ | ★★★★★ (Coder) | ★★★★★ |
| Multimodal | ★★★☆☆ (text+image) | ★★★★☆ (Plus variant) | ☆☆☆☆☆ (text only) |
| Multilingual | ★★★☆☆ (140+) | ★★★★★ (201+) | ★★★☆☆ |
| Edge Deployment | ★★★★★ (E2B/E4B) | ★★☆☆☆ | ★☆☆☆☆ |
| API Cost | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| Commercial License | ★★★★★ (Apache 2.0) | ★★★★★ (Apache 2.0) | ★★★★★ (MIT) |
| Community Ecosystem | ★★★★☆ | ★★★☆☆ | ★★★★☆ |

### Quick Selection Guide

| Your Scenario | First Choice | Reason |
|--------------|-------------|--------|
| Phone/edge local inference | **Gemma 4 E2B/E4B** | Only frontier model that runs smoothly on phones |
| Strongest model on consumer GPU | **Gemma 4 26B MoE** | 3.8B active, runs on RTX 4090 |
| Strongest open-source reasoning | **Qwen 3.7 Max** | GPQA 92.4%, highest in open-source |
| AI coding assistant | **Qwen 3-Coder** | Purpose-built for agentic coding |
| Massive low-cost API | **DeepSeek V4 Flash** | $0.14/MTok, unbeatable |
| Math/science reasoning | **DeepSeek R1** | AIME 99.4%, near theoretical ceiling |
| Multilingual app (non-English) | **Qwen 3.5** | 201 language coverage |
| Commercial product (license certainty) | **Gemma 4 / DeepSeek** | Apache 2.0 / MIT, most permissive |
| Need image understanding | **Qwen 3.7 Plus** | Most mature multimodal in open-source |

---

## 5. Practical: How to Get Started?

### 5.1 Local Deployment

```bash
# Gemma 4 — simplest local experience
ollama run gemma4

# Qwen 3 — consumer GPU-friendly version
ollama run qwen3:30b-a3b

# DeepSeek V4 — via API (self-deployment barrier too high)
# Or use distilled R1
ollama run deepseek-r1:70b
```

### 5.2 API Access

All three model families offer official APIs or access through third-party platforms:

| Platform | Gemma 4 | Qwen 3 | DeepSeek V4 |
|----------|---------|--------|-------------|
| Official API | Google AI Studio | Alibaba Cloud DashScope | DeepSeek Platform |
| OpenRouter | ✅ | ✅ | ✅ |
| Self-deploy | Ollama / vLLM | vLLM / SGLang | vLLM (requires many GPUs) |

### 5.3 Fine-tuning

```python
# All three model families support LoRA/QLoRA fine-tuning
# Recommended tools: Unsloth (most efficient), PEFT, Axolotl

# Gemma 4 fine-tune example (Unsloth)
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained("google/gemma-4-31b")
# ... standard fine-tune flow
```

---

## 6. Common Pitfalls

### ❌ "Open-source models are all about the same, just pick any"

Far from it. Gemma 4 crushes the other two on edge deployment, Qwen 3-Coder crushes on coding, DeepSeek R1 crushes on math reasoning. Wrong selection can lead to 10x cost differences or massive capability gaps.

### ❌ "DeepSeek is cheap because it's low quality"

No. DeepSeek's low price comes from aggressive MoE design, FP8 training, and multi-token prediction. V4 Pro's 80.6% on SWE-bench proves it's no "cheap substitute."

### ❌ "The biggest model is always the best"

Qwen 3-Coder 480B is strongest at coding, but Qwen 3-30B-A3B still delivers a great experience on consumer GPUs. **Deployment constraints (VRAM, latency, cost) are usually more important than benchmark scores**.

---

## 7. Further Reading

### Technical Reports
1. [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4) — Official model card and evaluation details
2. [Qwen 3 Technical Report](https://arxiv.org/abs/2503.09965) — Architecture and training methodology
3. [DeepSeek V4 Technical Report](https://arxiv.org/abs/2502.04872) — Engineering breakthroughs in trillion-scale MoE
4. [DeepSeek R1 Paper](https://arxiv.org/abs/2501.12948) — Low-cost training of reasoning models

### Model Comparisons
1. [Artificial Analysis — Open Model Leaderboard](https://artificialanalysis.ai/) — Real-time open-source rankings
2. [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
3. [OpenCompass](https://opencompass.org.cn/) — Comprehensive evaluation maintained by the Chinese community

---

## Reflection Questions

1. If you could only pick one open-source model as your primary for the next year, would you choose Gemma 4, Qwen 3, or DeepSeek V4? Why?
2. DeepSeek V4 Pro and Claude Opus 4.6 are nearly identical on SWE-bench (80.6% vs 80.8%), but differ 35-100x in price. When is this price difference justified?
3. Why can Gemma 4 run on phones while Qwen and DeepSeek can't? Analyze from an architecture design perspective.

---

## Summary

| Model Family | One-liner | Strongest Dimension | Weakest Dimension |
|-------------|-----------|--------------------|-------------------|
| Gemma 4 | Open AI from pocket to data center | Edge deployment, commercial license | Absolute reasoning power |
| Qwen 3 | The open-source all-rounder | General reasoning, coding, multilingual | High deployment barrier, complex variants |
| DeepSeek V4 | Reasoning peak + price slasher | Math reasoning, API cost | Multimodal, latency |

**Key takeaway**: Top open-source models in 2026 are no longer "cheap alternatives" to closed-source — each is the globally strongest in specific dimensions, including against closed-source models. Understanding each model's design philosophy and strength boundaries is the key to making the right technical selection. Choosing a model isn't about picking the highest rank — it's about picking the best fit for your engineering constraints.

---

*Day 47 of 60 | LLM Fundamentals*
*Word count: ~3500 | Reading time: ~18 minutes*
