# LLM 基础知识教学大纲 v3

> **基于 Google Trends 数据优化**  
> **目标读者**: CS 毕业多年、想系统了解 LLM 领域的从业者  
> **形式**: 每日一篇（中英双语），15-20 分钟阅读量  
> **周期**: 12 周（60 篇，工作日更新）

---

## 热度优先级（基于 Google Trends 2026-03）

| 话题 | 热度 | 课程安排 |
|------|------|----------|
| 🔥🔥🔥 AI Agent | 爆发增长 | Week 7-9 核心 |
| 🔥🔥🔥 What is LLM | 持续高位 | Week 1-2 基础 |
| 🔥🔥 Attention-Free (Mamba) | +350% | Week 6 新增 |
| 🔥🔥 Agentic RAG | +300% | Week 8 新增 |
| 🔥🔥 Transformer | 稳定增长 | Week 1 重点 |
| 🔥🔥 MCP | 飙升新词 | Week 8 新增 |
| 🔥🔥 Google ADK | 飙升新词 | Week 8 新增 |
| 🔥 RAG | 高位略回落 | Week 7 |
| 🔥 RLHF | 尖峰后稳定 | Week 3 |
| 🔥 OpenClaw | 有搜索量 | Week 12 |
| 🔥 Vibe Coding | 飙升新词 | Week 11 新增 |

---

## 第一阶段：基础原理 (Week 1-3)

### Week 1: 从神经网络到 Transformer 🔥🔥

| Day | 主题 | 核心问题 | 文章链接 |
|-----|------|----------|----------|
| D1 | 神经网络速览 | 为什么深度学习能 work？ | [EN](articles/en/day01-neural-network-overview.md) / [中文](articles/zh/day01-neural-network-overview.md) |
| D2 | RNN 的兴衰 | 序列建模的第一次尝试 | [EN](articles/en/day02-rise-and-fall-of-rnn.md) / [中文](articles/zh/day02-rise-and-fall-of-rnn.md) |
| D3 | **Attention 的诞生** | "Attention is All You Need" 精读 | [EN](articles/en/day03-birth-of-attention.md) / [中文](articles/zh/day03-birth-of-attention.md) |
| D4 | **Transformer 架构详解** | Self-attention、Multi-head、Position encoding | [EN](articles/en/day04-transformer-architecture.md) / [中文](articles/zh/day04-transformer-architecture.md) |
| D5 | Encoder-Decoder 到 Decoder-only | BERT vs GPT，为什么 GPT 胜出 | [EN](articles/en/day05-encoder-decoder-to-decoder-only.md) / [中文](articles/zh/day05-encoder-decoder-to-decoder-only.md) |

**D3 附加材料**：
- 论文原文：[Attention is All You Need](https://arxiv.org/abs/1706.03762)
- Harvard Annotated Transformer：[The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

### Week 2: 语言模型的本质 🔥🔥🔥

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D6 | **什么是 LLM** | P(next_token\|context) —— 一句话定义 | 🔥🔥 what is LLM +100 |
| D7 | Tokenization | BPE、WordPiece，为什么 tokenizer 很重要 | token 相关搜索 |
| D8 | Embedding 的魔法 | 词向量、位置编码、语义空间 | 技术细节 |
| D9 | **Scaling Laws** | Chinchilla、Kaplan，为什么"大力出奇迹" | 模型规模问题 |
| D10 | 涌现能力 | Emergence —— 量变到质变，还是测量偏差？ | 争议话题 |

### Week 3: 训练范式 🔥

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D11 | Pre-training | 自监督学习，用互联网训练世界模型 | 训练基础 |
| D12 | Fine-tuning | 参数高效微调 (LoRA, QLoRA) | 实用技术 |
| D13 | **RLHF 详解** | 人类反馈强化学习，ChatGPT 的秘密 | 🔥 rlhf +140% training |
| D14 | **RLHF 数据与实践** | 数据集构建、标注流程、成本 | 🔥 data +200% |
| D15 | DPO 与替代方案 | 为什么大家开始抛弃 PPO | 最新进展 |

---

## 第二阶段：深入理解 (Week 4-6)

### Week 4: 推理与生成

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D16 | 采样策略 | Temperature、Top-k、Top-p、Beam search | 常见参数 |
| D17 | KV Cache | 为什么推理可以加速，内存与计算 trade-off | 推理优化 |
| D18 | Speculative Decoding | 用小模型加速大模型推理 | 前沿技术 |
| D19 | Context Window | 从 2K 到 1M，长上下文技术演进 | 上下文限制 |
| D20 | **Prompt Engineering** | In-context learning，为什么提示词重要 | 实用技能 |

### Week 5: 模型能力边界

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D21 | **幻觉问题** | 为什么 LLM 会一本正经地胡说八道 | 🔥 常见问题 |
| D22 | 推理能力 | LLM 真的会推理吗？Chain-of-Thought 本质 | 能力争议 |
| D23 | 数学与逻辑 | 为什么算不好数学，Symbolic AI 回归 | 已知缺陷 |
| D24 | 世界模型 | LLM 有世界模型吗？Yann LeCun vs Ilya | 学术争论 |
| D25 | 评测与 Benchmark | MMLU、HumanEval、ARC —— 我们在测什么 | 评测方法 |

### Week 6: 模型架构进阶 🔥🔥

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D26 | **Mixture of Experts (MoE)** | 稀疏激活，用更少算力获得更大模型 | 架构创新 |
| D27 | Knowledge Distillation | 大模型→小模型，知识如何迁移 | 模型压缩 |
| D28 | **FlashAttention & Sparse Attention** | 内存带宽瓶颈，让 Transformer 跑更快 | 🔥 sparse attention +30% |
| D29 | **Attention-Free 架构** | Mamba、RWKV、State Space Models | 🔥🔥🔥 +350% 飙升 |
| D30 | 多模态基础 | CLIP、Vision Transformer、GPT-4V | 多模态入门 |

---

## 第三阶段：Agentic AI (Week 7-9) 🔥🔥🔥

### Week 7: Agent 基础

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D31 | **什么是 AI Agent** | 从 LLM 到 Agent，核心区别是什么 | 🔥🔥🔥 ai agent 爆发 |
| D32 | Agent 架构模式 | ReAct、Plan-and-Execute、自主循环 | 设计模式 |
| D33 | **Tool Use** | Function Calling，让 LLM 操作真实世界 | 核心能力 |
| D34 | Memory 系统 | 短期记忆、长期记忆、向量数据库 | 记忆设计 |
| D35 | **RAG 详解** | 检索增强生成，知识与推理结合 | 🔥 what is rag +70% |

### Week 8: Agent 进阶 🔥🔥 (新热词集中)

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D36 | Multi-Agent 系统 | 多 Agent 协作，涌现与混乱 | 架构模式 |
| D37 | **Agentic RAG** | RAG + Agent 结合，动态检索 | 🔥🔥 +300% 飙升 |
| D38 | **MCP (Model Context Protocol)** | Anthropic 的上下文协议标准 | 🔥🔥 飙升新词 |
| D39 | **Google ADK** | Agent Development Kit 介绍 | 🔥🔥 飙升新词 |
| D40 | Agent 工具对比 | OpenClaw vs Codex vs ADK vs LangChain | 🔥 best ai agent +400% |

### Week 9: Agent 瓶颈与前沿

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D41 | **可靠性问题** | 为什么 Agent 经常失败，错误累积 | 核心痛点 |
| D42 | 评估难题 | 如何评测 Agent？SWE-bench、WebArena | 评测方法 |
| D43 | 安全与对齐 | Agent 的安全风险，Prompt Injection | 安全问题 |
| D44 | 人机协作 | Human-in-the-loop，什么时候该介入 | 交互设计 |
| D45 | 成本与延迟 | Agent 太贵太慢，如何优化 | 实用问题 |

---

## 第四阶段：生态与实践 (Week 10-12)

### Week 10: 开发实践

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D46 | API 设计与选择 | OpenAI、Anthropic、Google —— 如何选 | 实用指南 |
| D47 | 开源 vs 闭源 | Llama、Mistral，什么时候用开源 | 技术选型 |
| D48 | 本地部署 | Ollama、vLLM、量化推理 | 部署实践 |
| D49 | 评估与监控 | 如何知道你的 LLM 应用是否 work | 运维实践 |
| D50 | Prompt 管理 | 版本控制、A/B 测试、迭代优化 | 工程实践 |

### Week 11: 行业应用

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D51 | 对话与客服 | Chatbot 的正确打开方式 | 应用场景 |
| D52 | 内容生成 | 写作、营销、创意 —— AI 能替代人吗 | 应用场景 |
| D53 | 知识管理 | 企业知识库、文档问答 | 应用场景 |
| D54 | **开发者工具与 Vibe Coding** | Copilot、Cursor，AI 如何改变编程 | 🔥 vibe coding 飙升 |
| D55 | 研究与科学 | AlphaFold、AI for Science | 前沿应用 |

### Week 12: OpenClaw 实现原理 🔥

| Day | 主题 | 核心问题 | 热度来源 |
|-----|------|----------|----------|
| D56 | **OpenClaw 架构总览** | Gateway、Session、Channel | 🔥 openclaw 飙升 |
| D57 | 消息路由与多通道 | Telegram/Discord/Signal 统一接入 | 系统设计 |
| D58 | Tool 系统设计 | exec/browser/nodes —— 能力扩展 | 工具系统 |
| D59 | Memory 与 Context 管理 | Compaction、workspace、长期记忆 | 记忆系统 |
| D60 | 子 Agent 编排 | sessions_spawn、subagent 任务分解 | 编排系统 |

---

## 设计原则

### 每篇文章结构（15-20 分钟阅读量）

```
1. 开篇问题 (1 min)
   - 用 Google 热门搜索问题引入
   - 例：D6 用 "what is LLM" 作为开头

2. 核心概念 (8-10 min) ← 主体
   - 3-5 个关键点
   - 类比解释 + 技术细节平衡

3. 代码/图示 (3-4 min)
   - 伪代码或架构图
   - 帮助理解的可视化

4. 数学推导 [选读] (+5 min)
   - 关键公式推导
   - 标注为选读，供数学强的同学

5. 常见误解 (2 min)
   - 澄清一个常见的错误认知

6. 延伸阅读 + 思考题 (1 min)
   - 2-3 篇推荐论文/文章
   - 一个开放性问题
```

**字数控制**: 2500-3500 字（不含选读数学部分）

### 难度曲线

```
Week 1-3:  ████░░░░░░ 入门级，重建基础认知
Week 4-6:  ██████░░░░ 进阶级，深入技术细节
Week 7-9:  ████████░░ 专业级，Agent 核心（热度最高）
Week 10-12: ██████████ 综合级，实践与应用
```

### 双语策略

- **中文版**: 主体内容，术语首次出现时标注英文
- **英文版**: 完整翻译，适合英文阅读习惯
- **术语表**: 维护中英术语对照表

---

## v3 相比 v2 的改动

| 改动 | 原因 |
|------|------|
| D3 添加论文精读链接 | transformer paper +100% |
| D28 扩展为 FlashAttention & Sparse Attention | sparse attention +30% |
| D29 新增 Attention-Free 架构 (Mamba/RWKV) | 🔥🔥🔥 +350% 飙升 |
| D30 调整为多模态基础（原多模态应用合并） | 为 D29 腾出位置 |

## v2 相比 v1 的改动

| 改动 | 原因 |
|------|------|
| D14 新增 "RLHF 数据与实践" | Google Trends: "data" +200% |
| D37 新增 "Agentic RAG" | Google Trends: +300% 飙升 |
| D38 新增 "MCP" | Google Trends: 飙升新词 |
| D39 新增 "Google ADK" | Google Trends: 飙升新词 |
| D40 改为 "Agent 工具对比" | Google Trends: "best ai agent" +400% |
| D54 加入 "Vibe Coding" | Google Trends: 飙升新词 |
| Week 12 改为 OpenClaw | Google Trends: "openclaw" 有搜索量 |

---

## 数据来源

- Google Trends (US, 过去 1 年, 2026-03-23 查询)
- 搜索词: "what is LLM", "how transformer works", "AI agent", "RAG", "RLHF", "transformer attention"

---

*v3 - 2026-03-23*
