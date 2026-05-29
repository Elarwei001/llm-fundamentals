# Day 41: Agent Reliability Engineering — 让 Agent 在生产环境中真正靠谱

> **核心问题**：为什么 Demo 里表现完美的 Agent，上了生产就频频翻车？如果你要给自己的 Agent 搭一套可靠性体系，第一步该做什么？

---

## 开篇

2023 年，Google 发表了 SRE（Site Reliability Engineering）这本书，核心思路是用工程方法解决运维可靠性问题——定义 SLO、设定 error budget、自动化缓解、blameless postmortem。二十年后回头看，SRE 已经是每个互联网公司的标配。

2025-2026 年，AI Agent 正在经历同样的转折。Demo 里 Agent 表现惊艳，但一上生产就遇到：多步骤流水线的错误复合、基础设施层面的 API 超时和 rate limit、上下文漂移导致指令被遗忘、以及 prompt injection 等安全问题。

今天这篇文章不只想告诉你"Agent 为什么会失败"——更重要的是：**如果你要给自己的 Agent 搭一套可靠性体系，具体该怎么做、用什么工具、怎么衡量做得好不好、怎么持续改进。**

---

## 1. Agent 为什么不可靠：复合可靠性问题

### 直觉：接力赛

把 Agent 流水线想象成一场接力赛。每个选手（步骤）都必须成功把接力棒传给下一个。任何一个选手掉棒——哪怕只有一次——比赛就输了。第三棒掉棒，不可能靠第五棒的出色发挥弥补。

对于一条由 **N** 个步骤组成的流水线，每步准确率为 **p**，端到端成功率是 p^N：

| 步骤数 | 每步 99% | 每步 95% | 每步 90% |
|-------|---------|---------|---------|
| 5     | 95.1%   | 77.4%   | 59.0%   |
| 10    | 90.4%   | 59.9%   | 34.9%   |
| 15    | 86.0%   | 46.3%   | 20.6%   |

![复合准确率曲线](./images/day41/compound-accuracy-chart.png)
*图 1：每步准确率在多步骤 Agent 流水线中的复合效应。*

关键洞察：**即使是"不错"的每步准确率，经过步骤复合后也会产生不可接受的端到端失败率。** 而且实际步骤之间并非独立——第二步的错误往往会降低第三步的质量，使实际情况更糟。

这不是 Agent 的 bug——这是顺序系统的基本数学性质。知道了这一点，接下来的问题就是：怎么用工程手段应对。

---

## 2. ARE：Agent Reliability Engineering

### 从 SRE 到 ARE

Google 在 2003 年提出 SRE 时，解决的痛点是"服务经常挂，靠人工救火不可持续"。2026 年的 Agent 面临完全一样的处境。

| SRE 概念 | ARE 对应 |
|---------|----------|
| SLO + Error Budget | Agent 的可接受失败率阈值 |
| Monitoring + Observability | Agent tracing + 每步验证 |
| Auto-remediation | 护栏 + 自纠错循环 |
| Fast Rollback | 检查点 + 从已知好状态恢复 |
| Incident Postmortem | Agent 失败的归因分析 |
| On-call + Alerting | Agent 异常告警 + 升级机制 |

**Agent Engineering ≈ Software Engineering（怎么写好一个系统），ARE ≈ SRE（怎么让它在生产环境里稳定运行）。**

以下是一套从零开始搭建 ARE 的五步路径。

---

## 3. Step 1 — 可观测性：先看见问题

### 直觉：开车没有仪表盘

想象一辆没有速度表、没有油量表、没有故障灯的车。你能开——直到出问题的时候，你完全不知道发生了什么。Agent tracing 就是 Agent 的仪表盘。

### 最小可用的 tracing 方案

一个 Agent trace 至少要捕获：

| 信息 | 为什么需要 |
|------|-----------|
| 每步的输入和输出 | 定位哪一步出了问题 |
| 每步的 latency (P50/P95/P99) | 发现性能瓶颈 |
| Token 使用量 | 控制成本 |
| 工具调用结果（成功/失败） | 区分模型错误 vs 基础设施错误 |
| 错误类型和消息 | 分类失败模式 |

### 工具选择

| 工具 | 特点 | 适合场景 |
|------|------|----------|
| **[Langfuse](https://langfuse.com)** | 开源，tracing + prompt 管理 + eval 一体化 | 中小团队、快速起步 |
| **[Datadog LLM Observability](https://www.datadoghq.com/product/ai/llm-observability/)** | 与现有 Datadog 基础设施深度集成，auto-instrumentation | 已在用 Datadog 的团队 |
| **[Arize Phoenix](https://arize.com/phoenix/)** | 开源，本地部署，专注 LLM eval | 对数据隐私要求高的场景 |
| **[Braintrust](https://www.braintrust.dev/)** | eval + experiment 优先 | 需要大量 A/B 实验的团队 |

**起步建议**：如果团队没有现有的 observability 平台，从 Langfuse 开始——开源、免费起步、5 分钟集成。如果已经在用 Datadog，直接用它的 LLM Observability 模块，不用额外引入新工具。

### 集成方式

这里提一个关键协议：**OpenTelemetry（OTel）**。它是由 CNCF（管 Kubernetes 的同一个基金会）维护的开源可观测性标准，定义了统一的 tracing/metrics/logging 数据格式（OTLP 协议）。简单说，它就是可观测性领域的"USB-C 接口"——**写一次 instrument 代码，数据可以发给任何支持 OTLP 的 backend**，今天用 Langfuse，明天换 Datadog，不用改业务代码。

对于 LLM/Agent 场景，社区在 OTel 之上做了两个扩展：OpenLLMetry（追踪 LLM 调用的 prompt/completion/token/cost）和 OpenInference（AI inference 的 tracing 规范）。

```python
# Langfuse 最小集成示例
from langfuse.decorators import observe

@observe()
def my_agent(user_query):
    # Langfuse 自动追踪这个函数的输入、输出、latency、token 用量
    result = agent.run(user_query)
    return result
```

---

## 4. Step 2 — 定义 SLO：什么是"够可靠"

### 直觉：没有目标就无法改进

SRE 的第一个原则是：**先定义什么是"够好"，然后度量你是否达到了。** Agent 也一样。

### Agent 的 SLI（Service Level Indicator）

| SLI | 定义 | 推荐目标 |
|-----|------|----------|
| **任务完成率** | 端到端任务成功（无需人工干预）的比例 | >85% |
| **步骤成功率** | 关键步骤（如工具调用、数据检索）的成功率 | >97% |
| **P95 latency** | 95% 的请求在多长时间内完成 | 因场景而异 |
| **Token 效率** | 成功完成任务的平均 token 数 | 持续优化方向 |
| **自恢复率** | 出错后通过重试/自纠错恢复的比例 | >70% |

### Error Budget 怎么定

如果 SLO 是"任务完成率 >85%"，那 error budget 就是允许的 15% 失败空间。当实际失败率逼近 budget 边界时：

1. **剩余 budget >50%**：正常迭代，可以上线新功能
2. **剩余 budget 20-50%**：谨慎上线，加强监控
3. **剩余 budget <20%**：暂停新功能，集中精力修可靠性问题

这比"感觉 Agent 不太稳定"的主观判断要精确得多。

---

## 5. Step 3 — 护栏：在错误传播之前拦截

### Pre-LLM 和 Post-LLM 两层防护

**Pre-LLM guardrails** 在输入到达模型之前拦截：
- 输入验证：拒绝格式错误、过长的查询
- 策略执行：确保请求不违反使用策略
- Prompt injection 检测：识别可疑的对抗性输入

**Post-LLM guardrails** 在输出到达用户之前拦截：
- 输出格式验证
- 幻觉检测：将声明与检索上下文交叉验证
- 敏感信息过滤：防止 PII 泄露

最有效的模式：将 post-LLM guardrail 的失败作为**自纠错循环的反馈**——不是拒绝输出，而是把错误信息反馈给 Agent 重试。

### 工具选择

| 工具 | 特点 | 适合场景 |
|------|------|----------|
| **[NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)** (NVIDIA) | 开源，5 种 rail 类型（input/dialog/retrieval/execution/output），对话流管理 | 复杂对话场景、需要精细策略控制 |
| **[Guardrails AI](https://github.com/guardrails-ai/guardrails)** | 结构化输出验证，60+ 预置 validators，RAIL 规范 | 需要 JSON schema 严格合规的场景 |
| **[LLM Guard](https://github.com/protectai/llm-guard)** (Protect AI) | 开源，PII/toxicity/prompt injection 扫描，自托管 | 对数据隐私要求高 |
| **Datadog Guardrails** | 集成在 observability 平台内，开箱即用 | 已在用 Datadog 的团队 |

**起步建议**：先写最简单的 if-else 检查（输入长度、输出格式），再考虑引入专门的 guardrails 框架。不要一上来就过度工程化。

---

## 6. Step 4 — 重试与检查点：让失败可以恢复

### 指数退避重试

Datadog 2026 年 AI 工程现状报告显示：**所有 LLM API 调用中有 5% 报告了错误，其中 60% 是 rate limit。** 这些不是模型的问题——是基础设施问题，用指数退避就能解决：

```python
import asyncio, random

async def agent_call_with_retry(agent_fn, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await agent_fn()
        except (RateLimitError, TimeoutError) as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
    raise Exception(f"Agent call failed after {max_retries} retries")
```

### 检查点：游戏存档点

就像在打 Boss 之前保存游戏——如果 10 步中的第 8 步失败了，从第 7 步的检查点恢复，而不是从头开始。

```python
class AgentCheckpoint:
    def __init__(self, task_id):
        self.task_id = task_id
        self.steps_completed = []
        self.results = {}

    def save(self, step_name, result):
        self.steps_completed.append(step_name)
        self.results[step_name] = result
        self._persist()

    def get_last_checkpoint(self):
        if not self.steps_completed:
            return None, {}
        return self.steps_completed[-1], self.results
```

---

## 7. Step 5 — 自纠错与 Human-in-the-Loop

### 自纠错：Reflexion 模式

**Reflexion**（Shinn 等人，NeurIPS 2023）让 Agent 评估自己的输出并重试：

1. Agent 尝试一个任务
2. 评估器检查输出质量
3. 如果有缺陷，Agent 收到批评反馈
4. Agent 带着反馈重试

到 2026 年，这个范式已经扩展到 **Process Reward Models（PRMs）**——专门对 Agent 推理的每个中间步骤打分的模型，提供更细粒度的纠错反馈。

**关键局限**：自纠错帮不了那些**自信地犯错**的 Agent。如果 Agent 不知道自己错了，它就无法纠正。这就是为什么 guardrails 和外部验证仍然不可或缺。

### Human-in-the-Loop

对于高风险决策，让人类参与仍然是最可靠的模式：

- **审批关卡**：Agent 提出行动，人类审批后执行
- **置信度阈值**：Agent 不确定时升级给人类
- **抽样审查**：随机审查一部分 Agent 决策

关键在于选择合适的关卡——不是每一步都需要人类，但那些会产生不可逆后果的步骤需要。

---

## 8. 安全：AgentHarm 与 Prompt Injection

### Agent 的独特安全挑战

与真实世界交互的 Agent（浏览网页、处理邮件、操作文件系统）会暴露在对抗性输入面前。AgentHarm 基准（Andriushchenko 等人，ICLR 2025）是第一个专门针对多步骤 Agent 场景的安全基准：

- 110 个恶意任务（augmentation 后 440 个），覆盖 11 个 harm category（Fraud、Cybercrime、Harassment、Disinformation、Violence 等）
- 每个 malicious task 都有 benign counterpart，区分"模型 refuse 了"和"模型做不了"
- Scoring 通过 custom grading function + LLM judge 自动完成

**核心发现**：不加 jailbreak 时，GPT-4o-mini 的 HarmScore 就达 62.5–82.2%。应用 universal jailbreak template 后，Claude 3.5 Sonnet 的 HarmScore 从 13.5% 飙升到 68.7%。被 jailbreak 的模型智能不降——safety alignment 被绕过，但能力完好。

**关键启示**：Well-aligned model ≠ safe agent。单轮 chat 的 safety alignment 无法迁移到 multi-step tool-calling 场景。

---

## 9. 持续改进循环

有了以上五步，你的 ARE 体系应该跑起来了。但 ARE 不是一次性项目——它是一个持续改进循环：

```
定义 SLO → 度量实际表现 → 发现 gap → 分析根因 → 实施改进 → 验证 → 重新定义 SLO
```

### 改进优先级矩阵

| 问题类型 | 影响范围 | 修复成本 | 优先级 |
|---------|---------|---------|--------|
| Rate limit / timeout | 所有请求 | 低（加重试） | 🔴 立即修 |
| 幻觉级联 | 特定任务链 | 中（加 post-LLM guardrail） | 🔴 本周修 |
| 上下文漂移 | 长任务 | 中（加检查点） | 🟡 下个迭代 |
| Agent 选错工具 | 特定场景 | 高（改 skill 描述） | 🟡 持续优化 |
| 边缘 case 处理 | 少数请求 | 视情况 | 🟢 有空再修 |

### 度量改进效果

每一次改动后，用同一套 SLI 度量前后对比：

| 改动 | 预期影响 | 验证方法 |
|------|---------|---------|
| 加重试 | 步骤成功率 +10% | 对比 rate limit 错误恢复率 |
| 加 guardrail | 幻觉率 -50% | 抽样审查 + automated eval |
| 加检查点 | 恢复时间 -70% | 模拟第 N 步失败，测恢复耗时 |
| 优化 prompt | 任务完成率 +5% | A/B test on production traffic |

---

## 10. 总结

| 概念 | 一句话 |
|------|--------|
| **ARE** | Agent 的 SRE——用工程方法让 Agent 在生产环境里靠谱 |
| **可观测性** | Agent 的仪表盘——先看见问题才能解决问题 |
| **SLO** | 定义"够好"是多少——没有目标就无法改进 |
| **护栏** | 错误传播之前的拦截层 |
| **检查点** | 游戏存档点——失败后从最近的好状态恢复 |
| **Reflexion** | Agent 自我评估并重试 |
| **Human-in-the-Loop** | 高风险决策的最终安全网 |
| **AgentHarm** | Agent 安全评估的标杆基准 |

**核心要点**：Agent 可靠性不是模型问题，是系统问题。像对待生产服务一样对待你的 Agent——因为它是生产服务。从可观测性开始，定义 SLO，加护栏和重试，持续度量改进。这套方法在 SRE 领域已经被验证了二十年，现在轮到 Agent 了。

---

## 延伸阅读

### 基础论文
1. ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629)（Yao 等人，2023）
2. ["Reflexion: Language Agents with Verbal Reinforcement Learning"](https://arxiv.org/abs/2303.11366)（Shinn 等人，NeurIPS 2023）
3. ["AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents"](https://arxiv.org/abs/2410.09024)（Andriushchenko 等人，ICLR 2025）

### 行业报告与工具
4. ["State of AI Engineering 2026"](https://www.datadoghq.com/state-of-ai-engineering/)（Datadog）— 生产环境的 LLM API 错误率数据
5. ["State of Agent Engineering"](https://www.langchain.com/state-of-agent-engineering)（LangChain）— 57% 的组织已有 agent 在生产环境，质量是最大阻碍
6. [Langfuse](https://langfuse.com) — 开源 Agent tracing + eval 平台
7. [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) — NVIDIA 的开源 guardrails 框架

### SRE 经典（ARE 的方法论根基）
8. [*Site Reliability Engineering*](https://sre.google/sre-book/table-of-contents/)（Google SRE Team, 2016）— SLO、error budget、postmortem 的方法论源头

---

## 思考题

1. 如果你的 Agent 当前的任务完成率是 70%，你会先加可观测性还是先加护栏？为什么？
2. 一个 10 步的 Agent 流水线，你在第 3 步（工具调用）和第 8 步（生成最终报告）都发现了 10% 的失败率。你会优先修哪一个？为什么？
3. 你的 Agent 在 AgentHarm 基准上 HarmScore 很低（很安全），但用户反馈说它经常拒绝合理请求。你会怎么平衡安全性和可用性？

---

*Day 41 of 60 | LLM Fundamentals*
*字数：约 3000 | 阅读时间：约 15 分钟*
