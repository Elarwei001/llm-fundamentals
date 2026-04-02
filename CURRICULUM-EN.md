# LLM Fundamentals Curriculum v3

> **Optimized based on Google Trends data**  
> **Target audience**: CS graduates who want to systematically understand the LLM field  
> **Format**: One article per day (bilingual CN/EN), 15-20 minutes reading time  
> **Duration**: 12 weeks (60 articles, weekday updates)

---

## Topic Priority (Based on Google Trends March 2026)

| Topic | Popularity | Course Placement |
|-------|------------|------------------|
| 🔥🔥🔥 AI Agent | Explosive growth | Week 7-9 Core |
| 🔥🔥🔥 What is LLM | Sustained high | Week 1-2 Basics |
| 🔥🔥 Attention-Free (Mamba) | +350% | Week 6 New |
| 🔥🔥 Agentic RAG | +300% | Week 8 New |
| 🔥🔥 Transformer | Steady growth | Week 1 Focus |
| 🔥🔥 MCP | Rising new term | Week 8 New |
| 🔥🔥 Google ADK | Rising new term | Week 8 New |
| 🔥 RAG | High but declining | Week 7 |
| 🔥 RLHF | Stable after spike | Week 3 |
| 🔥 OpenClaw | Has search volume | Week 12 |
| 🔥 Vibe Coding | Rising new term | Week 11 New |

---

## Phase 1: Fundamentals (Week 1-3)

### Week 1: From Neural Networks to Transformer 🔥🔥

| Day | Topic | Core Question | Article |
|-----|-------|---------------|---------|
| D1 | Neural Network Overview | Why does deep learning work? | [EN](articles/en/day01-neural-network-overview.md) / [中文](articles/zh/day01-neural-network-overview.md) |
| D2 | Rise and Fall of RNN | First attempt at sequence modeling | [EN](articles/en/day02-rise-and-fall-of-rnn.md) / [中文](articles/zh/day02-rise-and-fall-of-rnn.md) |
| D3 | **Birth of Attention** | What does "Attention is All You Need" say? | [EN](articles/en/day03-birth-of-attention.md) / [中文](articles/zh/day03-birth-of-attention.md) |
| D4 | **Transformer Architecture** | Self-attention, Multi-head, Position encoding | [EN](articles/en/day04-transformer-architecture.md) / [中文](articles/zh/day04-transformer-architecture.md) |
| D5 | Encoder-Decoder to Decoder-only | BERT vs GPT, why GPT won | [EN](articles/en/day05-encoder-decoder-to-decoder-only.md) / [中文](articles/zh/day05-encoder-decoder-to-decoder-only.md) |

**D3 Supplementary Materials**:
- Original paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- Harvard Annotated Transformer: [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

### Week 2: The Essence of Language Models 🔥🔥🔥

| Day | Topic | Core Question | Article |
|-----|-------|---------------|---------|
| D6 | **What is an LLM** | P(next_token\|context) — One sentence definition | [EN](articles/en/day06-what-is-an-llm.md) / [中文](articles/zh/day06-what-is-an-llm.md) |
| D7 | Tokenization | BPE, WordPiece, why tokenizers matter | [EN](articles/en/day07-tokenization.md) / [中文](articles/zh/day07-tokenization.md) |
| D8 | The Magic of Embeddings | Word vectors, positional encoding, semantic space | [EN](articles/en/day08-the-magic-of-embeddings.md) / [中文](articles/zh/day08-the-magic-of-embeddings.md) |
| D9 | **Scaling Laws** | Chinchilla, Kaplan, why "bigger is better" | [EN](articles/en/day09-scaling-laws.md) / [中文](articles/zh/day09-scaling-laws.md) |
| D10 | Emergent Abilities | Emergence — Phase transition or measurement bias? | Controversial topic |

### Week 3: Training Paradigms 🔥

| Day | Topic | Core Question | Trend Source |
|-----|-------|---------------|--------------|
| D11 | Pre-training | Self-supervised learning, training world models on the internet | Training basics |
| D12 | Fine-tuning | Parameter-efficient fine-tuning (LoRA, QLoRA) | Practical technique |
| D13 | **RLHF Explained** | Reinforcement Learning from Human Feedback, ChatGPT's secret | 🔥 rlhf training +140% |
| D14 | **RLHF Data & Practice** | Dataset construction, annotation workflow, costs | 🔥 data +200% |
| D15 | DPO and Alternatives | Why people are abandoning PPO | Latest developments |

---

## Phase 2: Deep Understanding (Week 4-6)

### Week 4: Inference and Generation

| Day | Topic | Core Question | Trend Source |
|-----|-------|---------------|--------------|
| D16 | Sampling Strategies | Temperature, Top-k, Top-p, Beam search | Common parameters |
| D17 | KV Cache | Why inference can be accelerated, memory-compute trade-off | Inference optimization |
| D18 | Speculative Decoding | Using small models to speed up large model inference | Cutting-edge technique |
| D19 | Context Window | From 2K to 1M, long context technology evolution | Context limitations |
| D20 | **Prompt Engineering** | In-context learning, why prompts matter | Practical skill |

### Week 5: Model Capability Boundaries

| Day | Topic | Core Question | Trend Source |
|-----|-------|---------------|--------------|
| D21 | **Hallucination Problem** | Why LLMs confidently make things up | 🔥 Common question |
| D22 | Reasoning Ability | Can LLMs really reason? The nature of Chain-of-Thought | Capability debate |
| D23 | Math and Logic | Why LLMs struggle with math, return of Symbolic AI | Known limitations |
| D24 | World Models | Do LLMs have world models? Yann LeCun vs Ilya | Academic debate |
| D25 | Evaluation & Benchmarks | MMLU, HumanEval, ARC — What are we measuring? | Evaluation methods |

### Week 6: Advanced Model Architecture 🔥🔥

| Day | Topic | Core Question | Trend Source |
|-----|-------|---------------|--------------|
| D26 | **Mixture of Experts (MoE)** | Sparse activation, more model with less compute | Architecture innovation |
| D27 | Knowledge Distillation | Large to small model, how knowledge transfers | Model compression |
| D28 | **FlashAttention & Sparse Attention** | Memory bandwidth bottleneck, making Transformer faster | 🔥 sparse attention +30% |
| D29 | **Attention-Free Architectures** | Mamba, RWKV, State Space Models | 🔥🔥🔥 +350% rising |
| D30 | Multimodal Basics | CLIP, Vision Transformer, GPT-4V | Multimodal intro |

---

## Phase 3: Agentic AI (Week 7-9) 🔥🔥🔥

### Week 7: Agent Basics

| Day | Topic | Core Question | Trend Source |
|-----|-------|---------------|--------------|
| D31 | **What is an AI Agent** | From LLM to Agent, what's the core difference? | 🔥🔥🔥 ai agent explosive |
| D32 | Agent Architecture Patterns | ReAct, Plan-and-Execute, autonomous loops | Design patterns |
| D33 | **Tool Use** | Function Calling, letting LLMs interact with the real world | Core capability |
| D34 | Memory Systems | Short-term, long-term memory, vector databases | Memory design |
| D35 | **RAG Explained** | Retrieval-Augmented Generation, combining knowledge with reasoning | 🔥 what is rag +70% |

### Week 8: Advanced Agents 🔥🔥 (New Hot Topics)

| Day | Topic | Core Question | Trend Source |
|-----|-------|---------------|--------------|
| D36 | Multi-Agent Systems | Multiple agents collaborating, emergence and chaos | Architecture patterns |
| D37 | **Agentic RAG** | RAG + Agent combination, dynamic retrieval | 🔥🔥 +300% rising |
| D38 | **MCP (Model Context Protocol)** | Anthropic's context protocol standard | 🔥🔥 rising new term |
| D39 | **Google ADK** | Agent Development Kit introduction | 🔥🔥 rising new term |
| D40 | Agent Tool Comparison | OpenClaw vs Codex vs ADK vs LangChain | 🔥 best ai agent +400% |

### Week 9: Agent Bottlenecks and Frontiers

| Day | Topic | Core Question | Trend Source |
|-----|-------|---------------|--------------|
| D41 | **Reliability Issues** | Why agents often fail, error accumulation | Core pain point |
| D42 | Evaluation Challenges | How to evaluate agents? SWE-bench, WebArena | Evaluation methods |
| D43 | Safety and Alignment | Agent security risks, Prompt Injection | Security issues |
| D44 | Human-AI Collaboration | Human-in-the-loop, when to intervene | Interaction design |
| D45 | Cost and Latency | Agents are too expensive and slow, how to optimize | Practical issues |

---

## Phase 4: Ecosystem and Practice (Week 10-12)

### Week 10: Development Practice

| Day | Topic | Core Question | Trend Source |
|-----|-------|---------------|--------------|
| D46 | API Design and Selection | OpenAI, Anthropic, Google — How to choose | Practical guide |
| D47 | Open Source vs Closed Source | Llama, Mistral, when to use open source | Technical choices |
| D48 | Local Deployment | Ollama, vLLM, quantized inference | Deployment practice |
| D49 | Evaluation and Monitoring | How to know if your LLM application works | Operations practice |
| D50 | Prompt Management | Version control, A/B testing, iterative optimization | Engineering practice |

### Week 11: Industry Applications

| Day | Topic | Core Question | Trend Source |
|-----|-------|---------------|--------------|
| D51 | Conversation and Customer Service | The right way to build chatbots | Use case |
| D52 | Content Generation | Writing, marketing, creativity — Can AI replace humans? | Use case |
| D53 | Knowledge Management | Enterprise knowledge bases, document Q&A | Use case |
| D54 | **Developer Tools & Vibe Coding** | Copilot, Cursor, how AI is changing programming | 🔥 vibe coding rising |
| D55 | Research and Science | AlphaFold, AI for Science | Frontier applications |

### Week 12: OpenClaw Implementation 🔥

| Day | Topic | Core Question | Trend Source |
|-----|-------|---------------|--------------|
| D56 | **OpenClaw Architecture Overview** | Gateway, Session, Channel | 🔥 openclaw rising |
| D57 | Message Routing and Multi-Channel | Unified access for Telegram/Discord/Signal | System design |
| D58 | Tool System Design | exec/browser/nodes — Capability extension | Tool system |
| D59 | Memory and Context Management | Compaction, workspace, long-term memory | Memory system |
| D60 | Sub-Agent Orchestration | sessions_spawn, subagent task decomposition | Orchestration system |

---

## Design Principles

### Article Structure (15-20 min reading time)

```
1. Opening Question (1 min)
   - Hook with a popular Google search question
   - Example: D6 opens with "what is LLM"

2. Core Concepts (8-10 min) ← Main body
   - 3-5 key points
   - Balance analogies with technical details

3. Code/Diagrams (3-4 min)
   - Pseudocode or architecture diagrams
   - Visuals to aid understanding

4. Math Derivation [Optional] (+5 min)
   - Key formula derivations
   - Marked as optional for math-strong readers

5. Common Misconceptions (2 min)
   - Debunk one common misunderstanding

6. Further Reading + Reflection (1 min)
   - 2-3 recommended papers/articles
   - One open-ended question
```

**Word count target**: 2500-3500 words (excluding optional math section)

### Difficulty Progression

```
Week 1-3:  ████░░░░░░ Beginner — Rebuild foundational understanding
Week 4-6:  ██████░░░░ Intermediate — Deep dive into technical details
Week 7-9:  ████████░░ Advanced — Agent core (highest popularity)
Week 10-12: ██████████ Comprehensive — Practice and applications
```

### Bilingual Strategy

- **Chinese version**: Primary content, English terms noted on first occurrence
- **English version**: Complete translation, adapted for English reading habits
- **Glossary**: Maintain a CN-EN terminology reference

---

## Changelog

### v3 Changes from v2

| Change | Reason |
|--------|--------|
| D3 added paper reading links | transformer paper +100% |
| D28 expanded to FlashAttention & Sparse Attention | sparse attention +30% |
| D29 added Attention-Free architectures (Mamba/RWKV) | 🔥🔥🔥 +350% rising |
| D30 adjusted to multimodal basics (merged multimodal apps) | Make room for D29 |

### v2 Changes from v1

| Change | Reason |
|--------|--------|
| D14 added "RLHF Data & Practice" | Google Trends: "data" +200% |
| D37 added "Agentic RAG" | Google Trends: +300% rising |
| D38 added "MCP" | Google Trends: rising new term |
| D39 added "Google ADK" | Google Trends: rising new term |
| D40 changed to "Agent Tool Comparison" | Google Trends: "best ai agent" +400% |
| D54 added "Vibe Coding" | Google Trends: rising new term |
| Week 12 changed to OpenClaw | Google Trends: "openclaw" has search volume |

---

## Data Sources

- Google Trends (US, past 1 year, queried 2026-03-23)
- Search terms: "what is LLM", "how transformer works", "AI agent", "RAG", "RLHF", "transformer attention"

---

*v3 - 2026-03-23*
