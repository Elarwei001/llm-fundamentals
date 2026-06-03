# Day 45: 成本与延迟 — 为什么你的 AI Agent 账单一直在涨

> **核心问题**：Token 价格自 2023 年以来下降了 99.7%，为什么你的 AI 账单反而涨了三倍？你又能做些什么？

---

你刚上线了第一个 AI Agent。它能读邮件、写回复、甚至安排会议。用户很喜欢。然后月度账单来了：$12,000。一个每天处理大约 500 个请求的工具。

你查看 token 定价。GPT-4.1 每百万输入 token 收 $5 — 仅为 2023 年 GPT-4 价格的一小部分。Gemini 2.5 Flash 不到 $0.30。Token 变便宜了。那钱花在哪了？

这就是 2026 年 AI Agent 的核心悖论：**token 更便宜了，账单更大了**。答案藏在「每个 token 的成本」和「每个任务的成本」之间的差距里。Agent 不会只做一次调用。它会循环、重试、检索、编排——最重要的是，它大量消耗的是模型周围的基础设施，而不是模型本身。

这篇文章拆解了 Agent 成本的真实来源、为什么延迟在多步工作流中会累积，以及生产团队用来削减 60–80% 账单的具体优化策略。

---

## 1. 隐性成本堆栈

#### 直觉：餐厅账单

把 AI Agent 想象成一顿餐厅饭菜。原材料（模型 token）可能很便宜，但账单包含了厨师的时间、厨房租金、服务员、预订系统和装修。你付的不仅仅是面粉——你付的是把面粉变成面包的整个运营。同样，你的 Agent 不仅仅为 token 买单——它为检索系统、重试循环、监控、等待下一个请求的空闲计算买单。

NavyaAI 2026 Token 成本报告精确量化了这一点：**模型推理仅占 Agent 总支出的 28%**。剩余的 72% 隐藏在大多数团队不追踪的基础设施、编排和运营中。

![Where Your AI Agent Budget Actually Goes](./images/day45/cost-breakdown-stack.png)
*图 1：AI Agent 的隐性成本堆栈。只有 28% 的总支出用于模型推理——其余的消失在编排、检索、可观测性和基础设施中。*

具体拆解如下：

| 成本类别 | 占比 | 包含内容 |
|---------|------|---------|
| 模型推理 | 28% | 输入 + 输出 token，嵌入生成 |
| 编排与重试 | 18% | Agent 循环控制，重试逻辑，降级链 |
| 检索与向量数据库 | 15% | 嵌入查询，分块，向量存储操作 |
| 基础设施与空闲 | 14% | 过度配置的 GPU 容量，冷启动 |
| 人工运维 | 13% | 事件响应，Prompt 维护，评估 |
| 可观测性与安全护栏 | 12% | 日志记录，内容过滤，输出验证 |

关键洞察：仅仅优化模型账单（选择更便宜的模型、减少 token 数量）最多只影响总成本的 28%。真正的节省来自优化整个堆栈。

---

## 2. 为什么 Agent 成本会爆炸

### 2.1 Agent 乘数效应

单次聊天机器人调用可能消耗 500 个输入 token 和 200 个输出 token。但执行同样任务的 Agent？它可能需要 15 次 LLM 调用：

1. 解析用户意图（1 次调用）
2. 检索相关文档（1 次嵌入调用 + 1 次生成调用）
3. 决定使用哪个工具（1 次调用）
4. 执行工具并解释结果（1 次调用）
5. 验证输出（1 次调用）
6. 处理格式错误后的重试（2 次调用）
7. 为用户总结和格式化（1 次调用）

每一步都携带自己的上下文——通常包括完整的对话历史。到第 7 步时，即使用户只输入了 50 个词，输入可能已经达到 4,000 个 token。

#### 直觉：电话会议

想象你问同事一个简单的问题：「我们 Q3 营收是多少？」在聊天机器人世界里，你得到一个答案。在 Agent 世界里，你的同事先查邮件，然后打开财务系统，再跟财务确认，最后格式化成漂亮的摘要。每一步都增加了时间和成本。一个简单的问题变成了数小时的过程。

### 2.2 上下文窗口膨胀

Agent 会累积上下文。一个 5 步 Agent 工作流加上工具结果，每步输入很容易超过 10,000 个 token。以 GPT-4.1 每百万输入 token $5 计算，每步 $0.05 — 乘以 5 步 = 每请求 $0.25。每天 10,000 个请求就是每天 $2,500，即一个 Agent **每月 $75,000**。

这还没算重试，重试可能让实际成本翻倍。生产环境的 Agent 通常有 10–30% 的重试率，原因包括格式错误、工具失败或幻觉检测。

### 2.3 延迟累积

![Agent Request Lifecycle & Latency Breakdown](./images/day45/agent-latency-breakdown.png)
*图 2：单次 Agent 步骤中时间的去向。LLM 推理占总延迟的约 70%。多步 Agent 将此乘以推理循环的步数。*

单次 Agent 步骤大约需要 2.5–3 秒：

| 阶段 | 典型延迟 | 备注 |
|------|---------|------|
| 意图解析 | ~50ms | 快但会累积 |
| 上下文组装 | ~200ms | RAG 检索，记忆查找 |
| Prompt 构建 | ~20ms | 模板渲染 |
| **LLM 推理** | **~2,000ms** | TTFT + token 生成 |
| 工具执行 | ~500ms | API 调用，代码执行 |
| 响应验证 | ~100ms | 安全护栏，格式检查 |

5 步 Agent 需要 12–15 秒。10 步 Agent 需要 25–30 秒。用户会注意到任何超过 5 秒的延迟。这迫使开发者使用更快（通常也更贵）的模型，形成成本-延迟的张力。

---

## 3. 模型定价全景

理解模型定价是成本优化的基础。各模型间的价格差距巨大——从最便宜到最昂贵大约有 **70 倍**。

![Model Pricing Landscape](./images/day45/model-pricing-landscape.png)
*图 3：主要模型的每美元输入 token 数量（2026 年 6 月）。70 倍的价格范围意味着模型选择是最大的成本杠杆。*

当前定价的关键观察（2026 年 6 月）：

| 模型 | 输入价格 ($/M tokens) | 输出价格 ($/M tokens) | 最适用场景 |
|------|---------------------|---------------------|----------|
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | 分类，提取 |
| Gemini 2.5 Flash | $0.30 | $2.50 | 通用 Agent |
| GPT-4.1 Mini | $0.40 | $1.60 | 路由，摘要 |
| GPT-4.1 | $5.00 | $15.00 | 复杂推理 |
| Claude Sonnet 4 | $3.00 | $15.00 | 均衡性能 |
| Claude Opus 4 | $15.00 | $75.00 | 最难的任务 |

实操结论：**大多数 Agent 步骤不需要最贵的模型。** 意图解析、工具选择和响应格式化可以使用比复杂推理便宜 10–50 倍的模型。这个观察驱动了最有影响力的优化策略：模型路由。

---

## 4. 优化策略

并非所有优化都是等价的。有些是低投入高回报的轻松胜利；另一些需要大量工程投入。下图按成本影响、延迟影响和实施难度绘制了每个策略。

![Optimization Strategy Impact Map](./images/day45/optimization-strategy-map.png)
*图 4：每个策略按成本降低（x 轴）和延迟降低（y 轴）绘制。气泡大小表示实施难度。右上象限是最佳区域。*

### 4.1 缓存（最高 ROI，最低投入）

#### Prompt Caching（Provider 级别）

当多个请求共享相同的 prompt 前缀——系统指令、few-shot 示例、工具定义——OpenAI 和 Anthropic 等提供商会自动缓存这些 token。OpenAI 的 prompt caching 可节省 **50% 的缓存输入 token 费用**。Anthropic 的前缀缓存对长 prompt 可提供高达 **90% 的成本降低**。

对 Agent 来说，这意义重大。每次 Agent 请求通常包含相同的系统 prompt（定义 Agent 角色、可用工具和行为规则），可能有 1,000–3,000 个 token。有了缓存，你只需在每个缓存窗口（通常 5–10 分钟）内为这些 token 付一次费。

**实施方法**：将静态内容放在 prompt 开头，动态内容放在末尾。大多数提供商会自动处理缓存。

#### Semantic Caching

Prompt caching 仅适用于精确的前缀匹配。Semantic caching 更进一步：它对传入的查询进行嵌入，检查之前是否回答过类似的查询，如果相似度超过阈值就返回缓存的响应。

威斯康星大学麦迪逊分校研究人员 2025 年的论文 ["Semantic Caching for Low-Cost LLM Serving"](https://arxiv.org/abs/2508.07675)（2025 年 8 月）将这种方法形式化，展示了自适应 semantic caching 可以将推理成本降低 60%，且对重复工作负载的质量影响极小。

[GPTCache](https://github.com/zilliztech/GPTCache) 和基于 Redis 的 semantic cache 等工具使得无需自定义 ML 基础设施即可使用此技术。

### 4.2 模型路由（最高成本节省）

#### 直觉：医院分诊

在急诊室里，不是每个病人都看资深外科医生。分诊护士评估严重程度——小伤口交给初级医生，复杂创伤升级到专家。模型路由的工作方式相同：轻量级分类器（甚至小 LLM）评估任务复杂度，路由到合适的模型。

2026 年 4 月的综述论文 ["Dynamic Model Routing and Cascading for Efficient LLM Inference"](https://arxiv.org/abs/2603.04445) 中的关键洞察是，仅通过智能路由，生产 RAG 系统就能实现 **27–55% 的成本降低**。

三种主要的路由模式：

| 模式 | 工作原理 | 成本节省 | 质量风险 |
|------|---------|---------|---------|
| 静态规则 | 基于正则/关键词的任务分类 | 30–40% | 低 |
| 语义路由 | 嵌入查询，按与已知类别的相似度分类 | 40–55% | 低–中 |
| LLM 作为路由器 | 小模型决定哪个模型处理请求 | 50–65% | 中 |

GitHub Copilot 使用的实际模式（在其 [2026 年 5 月关于 token 效率的博客文章](https://github.blog/ai-and-ml/github-copilot/improving-token-efficiency-in-github-agentic-workflows/)中描述）将文件导航任务路由到 Haiku，实现任务路由到 Sonnet，协调任务路由到 Opus——每步使用足够好的最便宜模型。

### 4.3 Prompt 压缩

[LLMLingua](https://arxiv.org/abs/2310.05736)（微软研究院，2023 年）及其后继者 [LongLLMLingua](https://arxiv.org/abs/2310.06839) 通过移除对输出贡献很小的 token 来压缩 prompt。一个小模型计算每个 token 的困惑度并移除低信息量的 token。

2–6 倍的压缩比很常见，且质量损失极小。对于传递 5,000 个 token 检索上下文的 Agent，压缩到 1,500 个 token 既节省成本又降低延迟。

2025 年 NAACL 综述 ["Prompt Compression for Large Language Models: A Survey"](https://aclanthology.org/2025.naacl-long.368.pdf) 提供了压缩技术及其权衡的全面概述。

### 4.4 批量处理

OpenAI 的 [Batch API](https://platform.openai.com/docs/guides/batch) 为 24 小时周转的异步请求提供 **50% 折扣**。对于任何不需要实时响应的 Agent 任务——夜间报告生成、批量文档处理、离线评估——这是白捡的钱。

**经验法则**：如果一个任务可以等 24 小时，就用 batch。仅此一项就能为有批量组件的工作负载削减 20–40% 的账单。

### 4.5 Speculative Decoding 和 Cascades

[Speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/)（在 Day 18 中讨论过）使用小型草稿模型预测 token，然后由大型模型并行验证。Google Research 2025 年 9 月的突破——[**Speculative Cascades**](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)——将此与模型级联结合：不再严格验证每个 token 与大型模型的匹配，而是使用灵活的延迟规则来决定小型模型的输出何时足够好。

这对 Agent 特别相关：许多 Agent 步骤（格式化、简单工具调用）不需要前沿模型的质量。Speculative cascades 让你用便宜模型处理简单的 70% token，只在困难的 30% 上升级到昂贵模型。

### 4.6 上下文窗口管理

每次调用都塞入 10,000 个 token 上下文的 Agent 在烧钱。策略包括：

- **滑动窗口**：只保留最近 N 轮对话
- **相关性过滤**：使用嵌入相似度只包含高于阈值的检索块
- **摘要压缩**：让小模型在传递给主模型前总结长上下文
- **Token 预算**：为每步上下文大小设定硬限制

Mem0 团队在其 [2026 Token 优化手册](https://mem0.ai/blog/the-2026-token-optimization-playbook-cut-ai-agent-memory-costs-3%E2%80%934x)中展示了结构化记忆架构可以将 Agent token 成本降低 3–4 倍，相比塞入完整对话历史。

---

## 5. 构建成本感知的 Agent 架构

#### 直觉：制造流水线

一个运营良好的工厂不会对每个步骤使用同一台机器。冲压用一种工具，焊接用另一种，喷漆又是一种。每台机器的规格与其任务匹配。成本感知的 Agent 架构遵循同样的原则：将模型、上下文大小和处理深度与每个步骤的实际需求匹配。

以下是一个实用的架构模式：

```
User Request
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│  Classifier  │────▶│  Route Decision  │
│  (Haiku/$0.25)│     │                  │
└─────────────┘     └────┬────────┬────┘
                         │        │
                    Simple      Complex
                         │        │
                         ▼        ▼
                   ┌─────────┐  ┌──────────┐
                   │  Flash  │  │  Sonnet  │
                   │  ($0.30) │  │  ($3.00)  │
                   └────┬────┘  └────┬─────┘
                        │            │
                        ▼            ▼
                   ┌──────────────────────┐
                   │  Semantic Cache      │
                   │  (check before call) │
                   └──────────┬───────────┘
                              │
                         Cache Miss?
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Compress Prompt     │
                   │  (LLMLingua, 2-3x)   │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  LLM Inference       │
                   │  (cached prefix)     │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Validate + Cache    │
                   │  (store for reuse)   │
                   └──────────────────────┘
```

这个架构按 ROI 顺序应用优化：先查缓存（命中则免费），然后路由到最便宜的足够模型，然后只在实际调用时才压缩。

---

## 6. 代码示例：带缓存的路由器

```python
import hashlib
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    input_price_per_m: float  # 每百万输入 token 的价格
    output_price_per_m: float
    max_context: int
    speed_tps: float  # 每秒 token 数

# 定义可用模型
MODELS = {
    "flash": ModelConfig("gemini-2.5-flash", 0.30, 2.50, 1_000_000, 150),
    "sonnet": ModelConfig("claude-sonnet-4", 3.00, 15.00, 200_000, 80),
    "gpt41": ModelConfig("gpt-4.1", 5.00, 15.00, 1_000_000, 60),
}

# 任务复杂度到模型的映射
TASK_MODEL_MAP = {
    "classify": "flash",      # 简单分类
    "extract": "flash",       # 信息提取
    "summarize": "flash",     # 摘要
    "route": "flash",         # 意图路由
    "reason": "sonnet",       # 多步推理
    "code": "gpt41",          # 代码生成
    "verify": "sonnet",       # 输出验证
}

class CostAwareRouter:
    """将请求路由到最便宜的足够模型，带缓存功能。"""
    
    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache: dict[str, str] = {}
        self.cache_ttl = cache_ttl_seconds
        self.stats = {"cache_hits": 0, "total_calls": 0, "cost_saved": 0.0}
    
    def _cache_key(self, prompt: str, task: str) -> str:
        """从 prompt + task 创建确定性的缓存键。"""
        content = f"{task}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def route(self, prompt: str, task: str, 
              force_model: Optional[str] = None) -> dict:
        """将请求路由到合适的模型。
        
        Args:
            prompt: 输入文本
            task: 任务类型 (classify, extract, reason 等)
            force_model: 覆盖路由，强制指定模型
            
        Returns:
            包含模型选择、预估成本和缓存状态的字典
        """
        self.stats["total_calls"] += 1
        
        # 第一步：检查缓存
        key = self._cache_key(prompt, task)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return {
                "model": "cache",
                "cached_response": self.cache[key],
                "cost": 0.0,
                "cache_hit": True,
            }
        
        # 第二步：选择模型
        model_name = force_model or TASK_MODEL_MAP.get(task, "sonnet")
        model = MODELS[model_name]
        
        # 第三步：预估成本
        input_tokens = len(prompt.split()) * 1.3  # 粗略估计
        estimated_cost = (input_tokens / 1_000_000) * model.input_price_per_m
        
        # 第四步：检查是否可以用更便宜的模型
        if task in ("classify", "extract", "summarize", "route"):
            if model_name not in ("flash",):
                cheaper = MODELS["flash"]
                savings = estimated_cost - (input_tokens / 1_000_000 * cheaper.input_price_per_m)
                self.stats["cost_saved"] += savings
        
        return {
            "model": model.name,
            "model_key": model_name,
            "estimated_cost": estimated_cost,
            "cache_hit": False,
            "input_tokens_est": int(input_tokens),
        }
    
    def store_response(self, prompt: str, task: str, response: str):
        """缓存响应以供将来复用。"""
        key = self._cache_key(prompt, task)
        self.cache[key] = response
    
    def get_stats(self) -> dict:
        """返回路由统计信息。"""
        hit_rate = (self.stats["cache_hits"] / max(self.stats["total_calls"], 1)) * 100
        return {
            **self.stats,
            "cache_hit_rate": f"{hit_rate:.1f}%",
        }


# 使用示例
router = CostAwareRouter()

# 简单分类 — 路由到最便宜的模型
result = router.route("Is this email urgent? 'Server is down, all users affected'", "classify")
print(f"Model: {result['model']}, Est. cost: ${result['estimated_cost']:.6f}")

# 复杂推理 — 使用更好的模型
result = router.route("Design a database schema for a multi-tenant SaaS platform", "reason")
print(f"Model: {result['model']}, Est. cost: ${result['estimated_cost']:.6f}")

# 同样的查询再次 — 缓存命中
router.store_response("Is this email urgent? 'Server is down'", "classify", "Yes, critical")
result = router.route("Is this email urgent? 'Server is down'", "classify")
print(f"Cache hit: {result['cache_hit']}, Cost: ${result['cost']:.6f}")

print(f"\nStats: {router.get_stats()}")
```

---

## 7. 常见误解

### ❌ 「直接用最便宜的模型就好了」

最便宜的模型（Gemini 2.5 Flash-Lite，$0.10/M 输入 token）按每个 token 省钱，但在复杂推理任务上可能失败，导致重试反而增加总成本。便宜模型上的一次失败尝试 + 一次成功尝试，可能比好模型上的一次成功尝试更贵。

**正确做法**：根据任务复杂度路由，而不是一刀切策略。

### ❌ 「Token 价格一直在降，成本不是真正的问题」

NavyaAI 2026 报告记录了这个悖论：**token 价格下降了 99.7%，但平均 AI 账单涨了三倍**。更便宜的 token 鼓励团队构建更大、更复杂的 Agent 工作流。消耗的 token 总量增长速度超过了每个 token 价格的下降速度。此外，60–80% 的真实成本完全在模型账单之外。

### ❌ 「延迟和成本是独立的问题」

它们深度耦合。慢的 Agent 促使开发者使用更大/更快的模型（增加成本），或者并行化步骤（增加总 token 消耗）。反过来，像模型路由这样的成本节省措施通常也能改善延迟——路由到更小的模型既便宜又快。

---

## 8. 度量与监控

你无法优化你没有度量的东西。生产级 Agent 系统需要在每个层面都有监控：

| 指标 | 追踪什么 | 目标 |
|------|---------|------|
| 每任务 token 数 | 每个完成的 Agent 任务的总输入 + 输出 token | 随时间递减 |
| 每任务成本 | 每个完成任务的美元成本（包括重试） | 简单任务 < $0.10 |
| 缓存命中率 | 从缓存提供服务的请求百分比 | 重复工作负载 > 40% |
| 重试率 | 失败并需要重试的 LLM 调用百分比 | < 15% |
| 完成时间 | 从用户请求到最终响应的墙上时钟时间 | 简单任务 < 5 秒 |
| 模型分布 | 路由到每个模型层的调用百分比 | 大部分是便宜模型 |

Gartner 2026 AI 成本管理研究发现，**实施实时 token 监控的团队在 60 天内将 AI 运营成本降低了 43%**——不是通过技术优化，而是通过推动更好设计决策的意识。

---

## 9. 前沿：接下来会发生什么

成本和延迟优化领域正在快速演进：

1. **编译式 Agent 工作流**（2026）：[Requesty.ai 2026 年 5 月的分析](https://www.requesty.ai/blog/ai-agent-techniques-may-2026-self-evolving-managed-compiled)描述了「工作流编译」——将稳定的 Agent 模式转换为微调的小模型。在 GPT-4.1 上每次运行成本 $0.05 的 5 步 Agent 工作流，可以编译成一次 Flash 微调调用，成本 $0.001，成本和延迟都降低 50 倍。

2. **自优化路由器**（2026）：路由正从静态规则转向基于强化学习的系统，持续根据观察到的质量和成本调整路由决策。[2026 年 4 月的路由综述](https://arxiv.org/abs/2603.04445)记录了主要部署中的这一转变。

3. **Speculative Cascades**（2025 年 9 月）：[Google Research 的混合方法](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)将 speculative decoding 与模型级联结合，使用灵活的延迟规则而非严格的 token 匹配，实现了比任一技术单独使用更好的成本-质量权衡。

4. **Agent Token 预算强制执行**（2026）：生产系统越来越多地实施每步硬性 token 预算，强制 Agent 在约束内工作，而不是消耗无限上下文。这正在成为 [AI.cc](https://natlawreview.com/press-releases/how-cut-ai-api-costs-80-aicc-publishes-step-step-token-optimization-guide) 等平台的核心实践，报告通过组合路由 + 压缩 + 输出长度控制实现 80% 的成本降低。

5. **多模态路由**（2026）：随着 Agent 处理文本、图像、音频和视频，路由决策必须考虑模态——将图像任务发送到视觉优化模型，音频发送到语音模型等。这为路由问题增加了新的维度，也为成本节省提供了新的机会。

---

## 10. 延伸阅读

### 实践指南
1. ["Techniques to Reduce AI Token Usage: The 2026 Playbook"](https://www.programstrategyhq.com/post/techniques-to-reduce-ai-token-usage-the-2026-playbook-for-cutting-costs-without-losing-quality) — 10 种技术的真实世界基准测试，2026 年 5 月
2. ["AI Agent Cost Optimization in 2026"](https://niteagent.com/blog/ai-agent-cost-optimization-2026/) — 多模型路由和缓存的实用模板，2026 年 5 月
3. ["LLM Token Optimization: Cut Costs & Latency"](https://redis.io/blog/llm-token-optimization-speed-up-apps/) — 基于 Redis 的缓存策略，2026 年 6 月
4. ["Improving Token Efficiency in GitHub Agentic Workflows"](https://github.blog/ai-and-ml/github-copilot/improving-token-efficiency-in-github-agentic-workflows/) — GitHub Copilot 的真实生产数据，2026 年 5 月

### 研究论文
1. ["Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey"](https://arxiv.org/abs/2603.04445) — 路由和级联方法的综合综述，2026 年 4 月
2. ["Semantic Caching for Low-Cost LLM Serving"](https://arxiv.org/abs/2508.07675) — 带在线学习的自适应 semantic caching，2025 年 8 月
3. ["Faster Cascades via Speculative Decoding"](https://arxiv.org/abs/2405.19261) — Google Research 的 speculative cascades 论文
4. ["LLMLingua: Compressing Prompts for Accelerated Inference"](https://arxiv.org/abs/2310.05736) — 微软的 prompt 压缩方法
5. ["Prompt Compression for Large Language Models: A Survey"](https://aclanthology.org/2025.naacl-long.368.pdf) — NAACL 2025 综合综述

### 报告
1. [NavyaAI AI Token Cost Report 2026](https://www.navyaai.com/reports/ai-cost-report-token-prices-vs-ai-bill) — 为什么 token 价格暴跌没有降低 AI 账单
2. [Mem0 Token Optimization Playbook 2026](https://mem0.ai/blog/the-2026-token-optimization-playbook-cut-ai-agent-memory-costs-3%E2%80%934x) — 记忆专项成本优化

---

## 思考题

1. 如果你要构建一个每天处理 10,000 个请求、平均每次请求 5 次 LLM 调用的 Agent，你会先实施哪个优化？为什么？

2. Token 价格每年大约下降 80%，但总 AI 支出持续上升。这说明单位成本和系统总成本之间是什么关系？这个趋势什么时候可能反转？

3. 你会如何设计一个路由系统来处理「便宜模型」5% 的时间产生微妙错误答案的情况？你会如何检测这种错误？

---

## 总结

| 概念 | 一句话解释 |
|------|----------|
| 隐性成本堆栈 | Agent 成本仅 28% 用于模型推理，其余在编排、检索和基础设施 |
| Agent 乘数效应 | Agent 每个用户请求需要 5–15 次 LLM 调用，每次上下文不断增长 |
| 模型路由 | 将任务路由到足够好的最便宜模型——27–55% 成本降低 |
| Prompt Caching | Provider 级别的缓存节省重复 prompt 前缀的 50–90% |
| Semantic Caching | 缓存相似查询（而非仅相同查询）——最高 60% 节省 |
| Prompt 压缩 | 移除低信息量 token——2–6 倍压缩，质量损失极小 |
| Batch API | 非紧急异步请求享受 50% 折扣 |
| Speculative Cascades | 级联 + speculative decoding 的混合，实现成本高效推理 |
| Token 预算 | 为每步上下文大小设定硬限制 |
| 成本监控 | 仅通过实时追踪就能将成本降低 43% |

**核心要点**：Token 价格暴跌了 99.7%，但 AI 账单涨了三倍，因为 Agent 在乘数循环中消耗 token。解决方案不是更便宜的 token——而是更聪明的架构。先查缓存，路由到最便宜的足够模型，压缩你发送的内容，度量一切。掌握这些的团队可以在不牺牲质量的情况下将 Agent 运行成本降低 60–80%。

---

*Day 45 of 60 | LLM Fundamentals*
*字数：约 3200 | 阅读时间：约 16 分钟*
