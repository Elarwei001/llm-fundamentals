# Day 46: API Design and Selection — Choosing the Right LLM Provider

> **Core Question**: With OpenAI, Anthropic, Google, DeepSeek, and others all offering LLM APIs, how do you pick the right one for your project — and how do you design your system so you're not locked in?

---

## Opening

Imagine you're building an AI-powered customer support tool. You need an LLM API. You open your browser and face a wall of options: OpenAI has GPT-5.4 and GPT-5.5, Anthropic offers Claude Opus 4.8 and Sonnet 4.6, Google has Gemini 3.5 Flash and 3.1 Pro, and DeepSeek and GLM undercut everyone at a fraction of the price.

Each provider claims to be the best. The benchmarks all look impressive. The pricing pages use different units, different tiers, and different caveats. Three months later, your API bill is 10x what you budgeted, or your model struggles with tasks you assumed it could handle.

The dirty secret of the LLM API market in 2026: **there is no single best provider**. The right choice depends on what you're building, how much latency you can tolerate, what your budget looks like, and whether you need multimodal input, structured output, or agentic tool use. This article gives you a systematic framework for making that decision.

---

## 1. The Provider Landscape (June 2026)

#### Intuition: Think of LLM providers like cloud providers

Just as AWS, Azure, and GCP each have strengths (AWS for ecosystem breadth, Azure for enterprise integration, GCP for data/AI), LLM providers have distinct profiles. You wouldn't migrate your entire company to a single cloud — and in 2026, the smartest teams don't commit to a single LLM provider either.

![Figure 1: LLM API Provider Landscape showing model hierarchies and pricing per million tokens](../zh/images/day46/provider-landscape.png)
*Figure 1: The LLM API provider landscape as of June 2026. Each provider offers a tiered model family, from budget-friendly to premium. Prices shown are per million tokens (input/output).*

### 1.1 OpenAI — The Ecosystem Leader

OpenAI maintains the broadest model catalog and the most mature developer tooling in 2026. Their model family spans five tiers:

| Model | Input (per MTok) | Output (per MTok) | Context Window | Best For |
|-------|-------------------|--------------------|----------------|----------|
| GPT-5.5 Pro | **$30.00** | **$180.00** | 1M | Hardest problems |
| GPT-5.5 | **$5.00** | **$30.00** | 1M | Complex reasoning, agents |
| GPT-5.4 | **$2.50** | **$15.00** | 1.1M | General-purpose production |
| GPT-5.4 Mini | **$0.75** | **$4.50** | 128K | Balanced cost/performance |
| GPT-5.4 Nano | **$0.20** | **$1.25** | 128K | High-volume, low-cost tasks |

Two API paradigms coexist in 2026:
- **Chat Completions API** — the industry-standard stateless interface. You send messages, get a response. Simple and widely compatible.
- **Responses API** — introduced in March 2025, recommended for new projects. Supports built-in tool use (web search, file search, code interpreter), server-side state management, and semantic streaming. OpenAI and Microsoft both recommend it as the default for new development ([OpenAI migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses)).

Key differentiators:
- **Structured outputs with JSON Schema validation** at the API level — the most mature implementation among providers
- **Prompt caching** saves up to 90% on repeated system prompts (cached input at $0.25/MTok for GPT-5.4)
- **Batch API** at 50% discount for non-urgent workloads
- **Realtime voice models** (May 2026): GPT-Realtime-2, GPT-Realtime-Translate, and GPT-Realtime-Whisper for voice applications ([OpenAI voice release](https://openai.com/index/advancing-voice-intelligence-with-new-models-in-the-api/))
- **GPT-5.5 Instant** (May 5, 2026): new ChatGPT default with 52.5% fewer hallucinated claims on high-stakes prompts covering medicine, law, and finance

### 1.2 Anthropic — The Agentic Coding Champion

Anthropic's Claude models excel at complex reasoning, code generation, and autonomous agent workflows. Their pricing saw a dramatic shift in 2026 with the Opus 4.6 launch in February — a 67% price reduction from the previous Opus 4.1 generation ($15/$75 down to $5/$25).

| Model | Input (per MTok) | Output (per MTok) | Context Window | Best For |
|-------|-------------------|--------------------|----------------|----------|
| Opus 4.8 | **$5.00** | **$25.00** | 1M | Complex agents, coding |
| Sonnet 4.6 | **$3.00** | **$15.00** | 1M | Balanced production |
| Haiku 4.5 | **$1.00** | **$5.00** | 1M | Fast, cost-effective |

Key differentiators:
- **Claude Opus 4.8** (May 28, 2026): improved coding, agentic tasks, and honesty — roughly 4x less likely than Opus 4.7 to let flaws in generated code pass unremarked ([Anthropic announcement](https://www.anthropic.com/news/claude-opus-4-8))
- **Long-context pricing simplified**: as of March 2026, Anthropic dropped long-context surcharges — all context lengths bill at flat rates
- **MCP (Model Context Protocol)**: native support for standardized tool/data source connections, open-sourced and gaining industry adoption
- **Prompt caching** up to 90% savings on repeated context; **batch processing** at 50% discount
- **Claude Code**: autonomous coding agent with terminal access, file operations, and multi-step workflows

Limitations to note: Anthropic did not offer native embeddings models or fine-tuning in early 2026, meaning teams needing those features may require a second provider.

### 1.3 Google — The Multimodal and Value Leader

Google's Gemini models offer the strongest price-performance ratio and the most capable multimodal support in 2026. In May 2026, Gemini 3.5 Flash launched, redefining the Flash series with reasoning capabilities that rival flagship models. Gemini 3.5 Pro is already in internal use and expected to roll out via API in July.

| Model | Input (per MTok) | Output (per MTok) | Context Window | Best For |
|-------|-------------------|--------------------|----------------|----------|
| 3.5 Flash | **$1.50** | **$9.00** | 1M | Frontier speed + quality |
| 3.1 Pro | **$2.00** | **$12.00** | 1M | General reasoning (>200K 2x) |
| 3 Flash | **$0.50** | **$3.00** | 1M | Fast production |
| 3.1 Flash-Lite | **$0.25** | **$1.50** | 1M | Ultra-budget |
| 2.5 Flash-Lite | **$0.10** | **$0.40** | 1M | Cheapest paid option |

Key differentiators:
- **Gemini 3.5 Flash** (May 19, 2026): reasoning capabilities rival flagship models, coding performance improved 10–20% over previous Flash generation while maintaining Flash-tier speeds ([Google announcement](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/))
- **Native video and audio understanding** — Gemini dominates multimodal long-context tasks
- **Context caching** saves approximately 90% on repeated inputs
- **Batch API** at 50% discount across all models
- **Free tier** available for development and testing — most generous among major providers
- **Vertex AI integration**: fine-tuning, deployment, monitoring in one Google Cloud stack
- **Gemini 3.1 Pro** (March 2026): cited as the "smartest default" by multiple reviewers, offering Claude-Opus-level quality at half the price

Limitations: function calling and structured output support are less mature than OpenAI's. Teams heavily dependent on tool-use JSON accuracy sometimes report more retries with Gemini compared to Claude or GPT.

### 1.4 DeepSeek — The Price Disruptor

DeepSeek, a Chinese AI lab focused on open-source, has emerged as the cost leader in 2026. Their OpenAI-compatible API makes switching remarkably easy.

| Model | Input (per MTok) | Output (per MTok) | Context Window | Best For |
|-------|-------------------|--------------------|----------------|----------|
| V4 Pro | **$0.435** | **$0.87** | 1M | Quality at low cost |
| V4 Flash | **$0.14** | **$0.28** | 1M | Cheapest capable option |

DeepSeek V4 Flash is roughly **35–100x cheaper** than GPT-5.5 or Claude Opus 4.8 at equivalent context lengths. For budget-sensitive applications — bulk classification, summarization, data extraction — DeepSeek often delivers adequate quality at a fraction of the cost ([DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)).

The trade-off: DeepSeek lacks ecosystem breadth — no multimodal support (text-only input), no native embeddings, limited fine-tuning, fewer enterprise features — and some teams report higher latency for real-time use cases.

### 1.5 Zhipu GLM — China's Open-Source Rising Star

Zhipu AI (Z.ai) is a leading Chinese AI lab whose GLM model series has become a noteworthy open-source option on the global market in 2026. GLM models are OpenAI-API-compatible and available to international developers through platforms like [OpenRouter](https://openrouter.ai/z-ai/glm-5.1).

| Model | Input (per MTok) | Output (per MTok) | Context Window | Best For |
|-------|-------------------|--------------------|----------------|----------|
| GLM-5.1 | **$0.98** | **$3.08** | ~200K | Complex reasoning, coding |
| GLM-5 | **$0.60** | **$1.92** | ~200K | General production |
| GLM-4.7 | **$0.40** | **$1.75** | ~131K | Balanced cost/performance |
| GLM-4.7-Flash | **Free** | **Free** | — | Development & testing |

Key differentiators:
- **GLM-5.1** (released March 27, 2026): major leap in coding capability, particularly for long-horizon tasks, positioned as comparable to Claude Opus tier
- **Open-source**: GLM models can be self-deployed, offering flexibility for teams with data sovereignty requirements
- **OpenAI-API-compatible**: migration cost is minimal — existing code only needs endpoint and API key changes
- **Coding Plan subscription**: Pro (~$30/mo) unlocks GLM-5, Max (~$80/mo) offers higher quotas
- **Free Flash models**: GLM-4.7-Flash and GLM-4.5-Flash are free for all registered users

Limitations to note: GLM models have shorter context windows (~200K) compared to the big three (1M); Zhipu raised API prices twice in early 2026 (+30% in February, +10% in April), narrowing the cost advantage; enterprise compliance certifications (SOC 2, HIPAA, etc.) are still in progress.

---

## 2. How to Compare Providers Beyond Price

Price per token is the most visible number, but it's rarely the most important one. Here are the dimensions that actually determine whether an API works for your application.

### 2.1 Total Cost of Ownership

The per-token rate is just the starting point. The real cost equation includes:

$$
\begin{aligned}
\text{Monthly Cost} &= \sum_{\text{requests}} \left( \text{tokens}_{\text{in}} \times \text{price}_{\text{in}} + \text{tokens}_{\text{out}} \times \text{price}_{\text{out}} \right) \\
&\quad + \text{retries} + \text{infrastructure} + \text{engineering time}
\end{aligned}
$$

Output tokens typically cost 5–6x more than input tokens across all providers. This means applications that generate long responses (code agents, document drafting) are far more expensive than applications that process large inputs but return short answers (classification, extraction).

![Figure 2: Monthly API cost comparison across providers and use cases](../zh/images/day46/cost-comparison-by-usecase.png)
*Figure 2: Estimated monthly cost for different use cases at 1,000 API calls/day. Notice the logarithmic scale — DeepSeek and Gemini Flash are orders of magnitude cheaper for high-volume tasks.*

Cost optimization levers:
- **Prompt caching** (all three major providers): up to 90% savings on repeated system prompts or context
- **Batch API** (all providers): 50% discount for non-urgent workloads processed asynchronously
- **Context window surcharges**: OpenAI doubles input token rates for GPT-5.4 beyond 272K tokens; Gemini charges 2x for contexts beyond 200K. Anthropic dropped surcharges in March 2026.
- **Model downgrading**: many applications can use a smaller model for 80%+ of requests and route only complex cases to premium models

### 2.2 Feature Maturity Matrix

Not all APIs are equal in how they implement key features. Here's a comparison of the capabilities that matter most for production systems:

| Feature | OpenAI | Anthropic | Google | DeepSeek | GLM |
|---------|--------|-----------|--------|----------|-----|
| Structured Output (JSON) | ★★★★★ Schema validation built-in | ★★★★☆ Reliable, well-formed JSON | ★★★☆☆ Improving, more retries needed | ★★★☆☆ Basic support | ★★★☆☆ OpenAI-compatible |
| Function Calling | ★★★★★ Parallel execution, strict mode | ★★★★☆ High accuracy, clean JSON | ★★★☆☆ Less mature | ★★★☆☆ OpenAI-compatible | ★★★☆☆ OpenAI-compatible |
| Streaming | ★★★★★ Semantic streaming (Responses API) | ★★★★☆ Reliable SSE streaming | ★★★★☆ Good SSE support | ★★★☆☆ Basic | ★★★☆☆ Basic |
| Prompt Caching | ★★★★★ Up to 90% savings | ★★★★★ Up to 90% savings | ★★★★★ ~90% savings | ★★☆☆☆ Limited | ★★☆☆☆ Limited |
| Embeddings | ★★★★★ Multiple models | ★☆☆☆☆ Not available natively | ★★★★★ Native embedding models | ★☆☆☆☆ Not available | ★★☆☆☆ Limited |
| Fine-tuning | ★★★★★ Supported | ★☆☆☆☆ Not available | ★★★★★ Via Vertex AI | ★★☆☆☆ Limited | ★★★☆☆ Open-source, self-hostable |
| Multimodal (Vision) | ★★★★☆ Strong | ★★★☆☆ Supported | ★★★★★ Native video/audio | ☆☆☆☆☆ Not supported | ★★☆☆☆ Basic |
| Enterprise Compliance | ★★★★★ SOC 2, HIPAA, ISO | ★★★★☆ SOC 2, expanding | ★★★★★ Full Google Cloud certs | ★★☆☆☆ Limited | ★★☆☆☆ In progress |

### 2.3 Latency and Reliability

Benchmarks don't tell the full latency story. What matters in production is:
- **Time to First Token (TTFT)**: how quickly the first token appears after sending a request
- **Tokens per Second (TPS)**: generation throughput after the first token
- **P99 latency**: the tail latency your users experience under load
- **Rate limits and throttling**: how the provider handles burst traffic

In 2026, the general latency hierarchy (fastest to slowest) for standard requests is approximately: DeepSeek V4 Flash ≈ Gemini Flash > Claude Haiku > GPT-5.4 Mini > Claude Sonnet > GPT-5.4 > Claude Opus > GPT-5.5. However, these vary significantly by region, payload size, and provider load. Always benchmark with your actual workload.

---

## 3. Provider Selection Framework

#### Intuition: Route by task, not by brand

Think of LLM APIs like a taxi fleet. You don't call a luxury sedan for every ride — you take an economy car for short trips and a premium vehicle for important client meetings. Multi-model routing works the same way.

![Figure 3: API selection decision tree for common use cases](../zh/images/day46/selection-decision-tree.png)
*Figure 3: A practical decision guide for choosing between providers based on your primary use case. Prices shown are per million tokens.*

### 3.1 Decision Principles

**Principle 1: Match model tier to task complexity.** Don't send a simple classification task to GPT-5.5 or Claude Opus. Use Nano, Flash-Lite, or Haiku for the 80% of requests that don't need frontier reasoning.

**Principle 2: Use multi-provider routing.** Tools like [LiteLLM](https://github.com/BerriAI/litellm) and [OpenRouter](https://openrouter.ai/) provide a unified interface across providers, letting you route requests by cost, latency, or capability without rewriting your application code.

**Principle 3: Design for switchability from day one.** Abstract your LLM calls behind an interface layer. Never hard-code provider-specific logic into your business logic. This costs almost nothing upfront and saves enormous pain when you need to migrate.

**Principle 4: Benchmark with your data.** Provider benchmarks measure performance on standardized datasets that may not reflect your domain. Run your own evaluation on 100–500 representative examples before committing.

### 3.2 Common Architectures

**Single provider, multiple tiers** — simplest to implement. Use one provider's full model family, routing by task complexity.

```
User Request → Router → GPT-5.4 Nano (simple)
                      → GPT-5.4 (medium)
                      → GPT-5.5 (complex)
```

**Multi-provider routing** — best cost optimization. Route each request to whichever provider offers the best price/quality tradeoff for that specific task type.

```
User Request → LiteLLM/OpenRouter → DeepSeek V4 Flash (classification)
                                 → Claude Sonnet 4.6 (agent tasks)
                                 → Gemini 3.1 Pro (multimodal)
```

**Fallback chain** — maximum reliability. Try your primary provider, automatically fail over to a secondary if it returns errors or exceeds latency thresholds.

```
User Request → Claude Sonnet → (timeout) → GPT-5.4 → (error) → Gemini Flash
```

![Figure 4: Capability profile comparison across providers — shapes show strengths, not overall ranking](../zh/images/day46/capability-radar.png)
*Figure 4: Radar charts illustrating how each provider has a distinct capability profile. The shape of the polygon matters more than its area — each provider has a unique "fingerprint" of strengths. These are illustrative scores for teaching purposes.*

---

## 4. Code Example: Building a Multi-Provider Router

Here's a practical example using LiteLLM to route requests across providers with automatic fallback:

```python
from litellm import completion
import os

# Configure API keys (set as environment variables)
# OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY

# Define model tiers for different task complexities
MODEL_TIERS = {
    "fast": "deepseek/deepseek-v4-flash",       # $0.14/$0.28 — cheapest
    "balanced": "gemini/gemini-2.5-flash",       # $0.30/$2.50 — good value
    "capable": "anthropic/claude-sonnet-4-6",    # $3.00/$15.00 — production quality
    "premium": "openai/gpt-5.5",                 # $5.00/$30.00 — best reasoning
}

def classify_complexity(prompt: str) -> str:
    """Simple heuristic to classify request complexity."""
    prompt_lower = prompt.lower()
    
    # Simple tasks: classification, extraction, short answers
    if any(w in prompt_lower for w in ["classify", "extract", "summarize", "is this"]):
        return "fast"
    
    # Medium tasks: writing, analysis, Q&A
    if any(w in prompt_lower for w in ["write", "analyze", "explain", "compare"]):
        return "balanced"
    
    # Complex tasks: multi-step reasoning, code generation
    if any(w in prompt_lower for w in ["debug", "implement", "plan", "reason"]):
        return "capable"
    
    # Default to balanced
    return "balanced"

def llm_call(prompt: str, tier: str = None) -> str:
    """Route an LLM call to the appropriate model tier."""
    tier = tier or classify_complexity(prompt)
    model = MODEL_TIERS[tier]
    
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback: if primary model fails, try GPT-5.4 as reliable backup
        print(f"Error with {model}: {e}. Falling back to GPT-5.4.")
        response = completion(
            model="openai/gpt-5.4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content

# Usage examples
result = llm_call("Classify this text as positive or negative: 'Great product!'")
# → Uses DeepSeek V4 Flash (fast tier) — ~$0.0001

result = llm_call("Implement a binary search tree with insert, delete, and search.")
# → Uses Claude Sonnet 4.6 (capable tier) — ~$0.05

result = llm_call("Solve this step by step: ...", tier="premium")
# → Uses GPT-5.5 (premium tier) — ~$0.10
```

The key architectural insight: **your application code never knows which provider it's talking to**. The routing logic is a thin layer that can be adjusted without touching business logic.

---

## 5. API Design Patterns for LLM Integration

Beyond provider selection, how you design your API integration matters as much as which provider you choose.

### 5.1 The Abstraction Layer

Always wrap your LLM calls in an interface. This pattern costs minutes to implement and saves weeks of migration effort:

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=kwargs.get("model", "gpt-5.4"),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

class AnthropicProvider(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=kwargs.get("model", "claude-sonnet-4-6"),
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
```

### 5.2 Cost Tracking

Build cost tracking into your abstraction layer from day one. Token counts and costs should be logged per request, per model, and per feature:

```python
import time

class TrackedLLMCall:
    def __init__(self, provider: LLMProvider, model: str):
        self.provider = provider
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def complete(self, prompt: str, **kwargs) -> str:
        start = time.time()
        response = self.provider.complete(prompt, model=self.model, **kwargs)
        latency = time.time() - start
        
        # Log metrics (in production, send to Prometheus/DataDog/etc.)
        print(f"[{self.model}] {latency:.2f}s | "
              f"tokens: ~{len(prompt.split())}in | cost: ~${self._estimate_cost(prompt):.4f}")
        return response

    def _estimate_cost(self, prompt: str) -> float:
        # Rough estimate; use actual token counts from API response in production
        PRICES = {
            "gpt-5.4": (2.50, 15.00),
            "claude-sonnet-4-6": (3.00, 15.00),
            "gemini-2.5-flash": (0.30, 2.50),
        }
        in_price, _ = PRICES.get(self.model, (1.0, 5.0))
        return len(prompt.split()) / 1_000_000 * in_price
```

---

## 6. Common Misconceptions

### ❌ "Just pick the cheapest provider"

Price per token is a starting point, not a decision. A model that's 10x cheaper but requires 3x more retries, produces lower-quality outputs, or lacks the features you need (structured output, tool calling) will cost more in engineering time and user frustration than you save on API bills.

### ❌ "Use one provider for everything"

Vendor lock-in with LLM APIs is particularly risky because the market is moving fast. A provider that's best today may not be best in six months. Multi-provider routing is both a cost optimization and a risk mitigation strategy.

### ❌ "Benchmarks tell you everything"

Provider benchmarks like MMLU, HumanEval, and SWE-bench are useful for understanding relative capability, but they don't reflect your specific domain, your prompt style, or your latency requirements. Always validate with your own data.

### ❌ "You need GPT-5.5 or Claude Opus for everything"

For the vast majority of production workloads — chatbots, content classification, summarization, extraction — models like GPT-5.4 Nano, Gemini Flash, or Claude Haiku deliver perfectly adequate quality at a fraction of the cost. Reserve premium models for the tasks that genuinely need them.

---

## 7. Frontier: What's Changing Fast

The LLM API landscape in 2026 is evolving rapidly. Here are the most significant recent developments:

| Development | Date | Impact |
|-------------|------|--------|
| [Claude Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8) release with dynamic workflows | May 28, 2026 | 4x better code honesty, same price as Opus 4.7 |
| [Gemini 3.5 Flash](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/) launch at **$1.50/9** per MTok | May 19, 2026 | Frontier-tier reasoning at Flash speeds, 10–20% coding gains |
| [GPT-5.5 Instant](https://openai.com/research/index/release/) as new ChatGPT default | May 5, 2026 | 52.5% fewer hallucinated claims on high-stakes prompts |
| [OpenAI Realtime Voice API](https://openai.com/index/advancing-voice-intelligence-with-new-models-in-the-api/) launch | May 7, 2026 | New class of voice apps with reasoning, translation, transcription |
| [GLM-5.1](https://openrouter.ai/z-ai/glm-5.1) launch at **$0.98/3.08** per MTok | Mar 27, 2026 | Open-source coding gains rivaling Claude Opus tier |
| [Anthropic dropped long-context surcharges](https://www.anthropic.com/claude/opus) | March 2026 | Flat pricing for all context lengths up to 1M tokens |
| [DeepSeek V4](https://api-docs.deepseek.com/quick_start/pricing) at **$0.14/0.28** per MTok | April 2026 | 35–100x cheaper than premium providers |
| [Gemini 3.1 Pro](https://ai.google.dev/) at **$2/12** per MTok | March 2026 | Claude-Opus-level quality at half the price |
| [Anthropic Mythos](https://www.reuters.com/business/anthropic-roll-out-claude-mythos-coming-weeks-launches-opus-48-2026-05-28/) announcement | May 28, 2026 | Next-gen model coming "in weeks," could shift market again |

The trend is clear: **prices are falling fast, capabilities are converging, and differentiation is shifting from raw model quality to ecosystem features** (tool calling, structured output, voice, multimodal, enterprise compliance).

---

## 8. Further Reading

### Documentation
1. [OpenAI API Documentation](https://developers.openai.com/api/docs) — complete API reference with guides
2. [Anthropic Claude API Documentation](https://docs.anthropic.com/en/docs) — Claude API guides and reference
3. [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs) — Gemini API guides and pricing
4. [DeepSeek API Documentation](https://api-docs.deepseek.com/) — DeepSeek API reference
5. [Zhipu GLM Open Platform](https://open.bigmodel.cn/) — GLM API guides and pricing

### Tools
1. [LiteLLM](https://github.com/BerriAI/litellm) — unified interface for 100+ LLM providers
2. [OpenRouter](https://openrouter.ai/) — API gateway routing across providers
3. [Instructor](https://python.useinstructor.com/) — structured output library for Python with multi-provider support

### Analysis
1. ["OpenAI vs Anthropic vs Google Cost Comparison" (LLM Gateway, 2026)](https://llmgateway.io/blog/openai-vs-anthropic-vs-google-cost-comparison)
2. ["Top LLM API Providers in 2026" (Fireworks AI)](https://fireworks.ai/blog/best-llm-api-providers)
3. ["LLM API Pricing Comparison 2026" (CloudZero)](https://www.cloudzero.com/blog/llm-api-pricing-comparison/)

---

## Reflection Questions

1. If your primary provider goes down for 4 hours tomorrow, how would your application behave? What would it cost you?
2. For your specific use case, what percentage of requests actually need a premium model? What would you save by routing the rest to cheaper alternatives?
3. How would you measure whether switching providers improved or worsened your application's quality? What metrics would you track?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| Multi-provider routing | Send each request to the cheapest model that can handle it well |
| Prompt caching | Reuse repeated context tokens at up to 90% discount |
| Batch API | Process non-urgent requests at 50% discount |
| LiteLLM / OpenRouter | Unified interfaces that abstract away provider differences |
| Abstraction layer | Wrap LLM calls in an interface so switching providers doesn't require rewriting application code |
| Cost of ownership | Per-token price × retries × engineering time × infrastructure = real cost |

**Key Takeaway**: In 2026, the winning strategy isn't picking the "best" LLM provider — it's building an architecture that lets you use the right model for each task, switch providers when the market shifts, and optimize cost without sacrificing quality. The best API call is the cheapest one that gets the job done reliably.

---

*Day 46 of 60 | LLM Fundamentals*
*Word count: ~2800 | Reading time: ~14 minutes*
