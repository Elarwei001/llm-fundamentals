# Day 60: 子 Agent 编排

> **核心问题**：AI 系统什么时候应该把工作拆给多个 sub-agent？orchestrator 又该如何让这个“小团队”变得可靠，而不是更混乱？

---

## 开场

想象一个资深工程师在指挥线上事故处理。一个人看日志，一个人检查刚上线的部署，一个人同步客服反馈，资深工程师负责让所有人盯住同一个目标。这样确实可能比一个人从头查到尾更快，但前提是这个“指挥者”真的在做管理：分配边界清晰的任务，避免重复劳动，要求证据，合并结论，并且在信息已经足够时让大家停下来。

Sub-agent orchestration 在 LLM 系统里做的就是这件事。主 agent，也就是 **orchestrator**，把有边界的子任务委托给专门的 sub-agent。每个 sub-agent 可以有自己的 context window、tools、model、sandbox、instructions 和 timeout。Orchestrator 决定要 spawn 什么 worker、传入哪些上下文、开放哪些工具、如何合并结果，以及什么时候继续协调已经不划算。

这是整门课的最后一篇。它把前面讲过的 agents、tools、memory、context management、evaluation、safety、cost、human-AI collaboration 串在一起。Sub-agent 不是“更多智能”的魔法按钮，而是一种工程模式，用来控制注意力、能力和风险。

---

## 1. 为什么需要 Sub-Agent

#### 直觉：餐厅厨房，而不是更大声的主厨

如果餐厅出菜太慢，对主厨喊得更大声没有用。真正能扩展的厨房会分工：烤炉、甜点、备菜、摆盘、质检。主厨不是每道菜都亲手做，而是决定先做什么、票据怎么流转、哪盘菜合格，以及如何避免厨房互相卡住。

早期 LLM agent 很像一个主厨试图在一个 context window 里做完所有事。短任务还能工作；一旦任务需要多种注意力，例如读代码库、测试行为、搜索新论文、生成图片、翻译、质量审查，问题就出现了。Context window 被无关证据塞满。某一步需要的 tool 权限留到了另一处不该使用的地方。Agent 变慢，也更容易被噪声带偏。

Sub-agent 出现，是因为长任务需要**有边界的工作间**。Research sub-agent 可以大范围阅读，不污染主上下文；test sub-agent 可以运行命令但不能改文件；reviewer sub-agent 可以批判 draft，而不用继承 writer 的假设；background sub-agent 可以等待慢 I/O，orchestrator 继续推进其他工作。

![图 1：Sub-agent orchestration lifecycle](./images/day60/subagent-orchestration-lifecycle.png)
*图 1：Orchestrator 先界定目标，再分解任务，spawn 有边界的 worker，合并输出，并记录 orchestration trace，便于审计和后续学习。*

[OpenAI Codex subagents](https://developers.openai.com/codex/subagents) 文档里，Codex 负责处理 spawn workers、路由 follow-up instructions、等待结果、合并回复等 orchestration 细节。[Spring AI 2026 年 1 月的 subagent orchestration pattern](https://spring.io/blog/2026/01/27/spring-ai-agentic-patterns-4-task-subagents/) 描述的是同一个核心模式：主 agent 通过 Task tool 委托任务，每个 sub-agent 在自己的 isolated context window 里运行。产品界面不同，但架构思想一致：**sub-agent 是被委托出去的执行上下文**。

关键是“委托”。Sub-agent 不只是又一次 API call。它应该收到局部任务、局部上下文、局部权限和返回契约。如果这些都很模糊，那就不是编排团队，而是在并行制造混乱。

---

## 2. Orchestrator 真正负责什么

#### 直觉：拿着分镜表的电影导演

电影导演不是简单地“多雇几台摄像机”。导演要决定拍哪场戏、每台机位拍什么、哪些 take 可用，以及最后怎么剪。没有分镜表，更多摄像机只会制造更多素材，不一定制造更好的电影。

Orchestrator 有五件核心工作：

| 责任 | 它回答的问题 | 如果忽略会怎样 |
|---|---|---|
| Decompose | 任务到底能拆成哪些子任务？ | Worker 解决重复或无关的问题 |
| Scope context | 每个 sub-agent 需要知道什么？ | Context 泄漏、注意力分散、遗漏约束 |
| Allocate capability | 该用哪个 model、tools、sandbox、timeout？ | 权限过大、工具不安全、成本浪费 |
| Aggregate | 结果如何合并，冲突如何解决？ | 矛盾输出被包装成流畅但错误的答案 |
| Stop | 什么时候不该继续 delegation？ | 无限循环、成本上涨、答案变慢 |

这个表故意比较的是职责，而不是产品。[Google ADK](https://adk.dev/)、[LangGraph](https://www.langchain.com/langgraph)、[AutoGen](https://microsoft.github.io/autogen/)、[CrewAI](https://www.crewai.com/)、OpenAI Codex subagents、Claude Code 风格 task agents、OpenClaw sessions 暴露的是不同 control surface。有的是 application framework，有的是 coding-agent 产品，有的是 personal-agent runtime。不能把它们压成一个“最佳 sub-agent 工具排行榜”。

在 OpenClaw 这类系统里，`sessions_spawn` 更适合被理解为一个 delegation boundary，而不是完整的 orchestration system。这个调用可以创建一个独立工作 session，但 parent 仍然要负责 contract：委托什么任务、哪些旧上下文可以安全传入、如何等待结果、如何处理失败、返回答案是否足够。这个区别很重要，因为 spawning 是一个 action；orchestration 是围绕这个 action 的 policy。

好的 orchestrator 往往比新手想象得更克制。它不应该 micromanage 每个 token，不应该把 parent conversation 整段贴给每个 worker，也不应该因为能并发就随手 spawn。它的职责是让 delegation 比单个拥挤上下文更便宜、更干净、更可靠。

---

## 3. 四种 Topology：Parallel、Sequential、Hierarchical、Hybrid

#### 直觉：不同 errands 需要不同路线

如果你要买菜、取干洗、拿快递，三个人可以并行跑。如果你要办护照，就不能先提交申请再去拍照。如果你还没搞清楚问题是什么，可能需要一个 supervisor 先调查，再决定怎么分工。任务依赖关系决定路线。

![图 2：Orchestration topology patterns](./images/day60/orchestration-topologies.png)
*图 2：Parallel、sequential、hierarchical、hybrid topology 对应不同依赖结构。Orchestrator 的第一个设计选择，就是工作图应该长什么样。*

| Topology | 最适合 | 注意风险 |
|---|---|---|
| Parallel fan-out | 子任务相互独立，结果可以合并 | 重复劳动、假设不一致 |
| Sequential pipeline | 每一步依赖上一步输出 | 关键路径慢，错误会向后传播 |
| Hierarchical supervisor-worker | 工作复杂，需要中心化路由 | Supervisor 成为瓶颈，过度 delegation |
| Hybrid plan-execute-verify | 既需要速度，也需要质量门禁 | 状态更多，调试更难 |

2026 年 2 月的论文 [AdaptOrch](https://arxiv.org/abs/2602.16873) 把这个直觉形式化了。它指出，当 frontier model 的 benchmark performance 越来越接近时，选择正确的 orchestration topology 可能比选择某个“最强 model”更重要。论文提出把 task decomposition graph 路由到 parallel、sequential、hierarchical 或 hybrid 模式，并在 coding、reasoning、retrieval-augmented generation 任务上报告了相对 static topology baseline 的 12-23% 提升。

2026 年 3 月的论文 [Benchmarking Multi-Agent LLM Architectures for Financial Document Processing](https://arxiv.org/abs/2603.22651) 在金融文档处理这个具体场景里提出了类似问题。它比较 sequential pipeline、parallel fan-out with merge、hierarchical supervisor-worker 和 reflexive self-correcting loop，用来做 financial document 的结构化抽取。重点不是“金融场景有一个万能赢家”，而是 orchestration 已经变成一个可衡量 cost-accuracy trade-off 的架构选择。

---

## 4. Minimal Sub-Agent Contract

#### 直觉：给同事发一张干净的工单

你让同事帮忙时，“你看一下这个”是一张很糟糕的工单。好的工单会写清楚你要什么结果、哪些文件或事实重要、对方能不能修改、什么时候要答案、返回格式是什么。Sub-agent 也需要这种局部契约。

![图 3：Minimal sub-agent contract](./images/day60/subagent-context-contract.png)
*图 3：Sub-agent 应该收到 scoped contract：objective、evidence、tools、return shape。把完整 parent context 丢过去，通常不如给一份精准 briefing。*

一个有用的 sub-agent contract 通常有四部分：

1. **Objective**：窄任务，写成 outcome，而不是模糊角色。
2. **Evidence**：它应该使用的具体文件、搜索结果、日志或约束。
3. **Capability**：允许使用的 tools、model 强度、network access、write access、timeout。
4. **Return shape**：orchestrator 期望收到的 summary、patch、JSON object、score 或 decision。

2026 年 2 月的论文 [AOrchestra](https://arxiv.org/abs/2602.03786) 把这个思路推进了一步。它把每个 agent 建模成一个 tuple：**Instruction、Context、Tools、Model**。这个 tuple 就像 on-demand specialization 的配方。系统不再只维护一组固定 static roles，而是让 orchestrator 为当前 subtask 创建 tailored executor。论文报告称，在搭配 Gemini-3-Flash 时，AOrchestra 在 GAIA、SWE-Bench 和 Terminal-Bench 上相对最强 baseline 有 16.28% 的相对提升。

这也直接连到 Day 59 的 memory 和 context management。Parent agent 可能知道很多东西，但 sub-agent 应该只收到它需要的部分。上下文太少，会盲干；上下文太多，会分心、泄漏信息，并形成隐性耦合。

---

## 5. Coordination Tax

#### 直觉：会议里多加人是有成本的

如果工作能清楚拆开，两个人可能比一个人更快。十个人挤在同一个会议里，问题反而可能更慢：更多同步、更多分歧、更多合并成本、更多“谁在做什么”的管理。Sub-agent 也有同样的 coordination tax。

![图 4：Coordination overhead curve](./images/day60/coordination-overhead-curve.png)
*图 4：增加 sub-agent 可能提高能力，但 coordination overhead 也会上升。实用区间通常是少量边界清晰的 worker。*

可以用一个简单的设计公式表达这个 trade-off：

$$
\begin{aligned}
\text{task value} &= \text{capability gain} + \text{parallelism gain} \\
&\quad - \text{coordination cost} - \text{merge risk} - \text{safety risk}
\end{aligned}
$$

这不是科学定律，而是提醒我们：“spawn more agents”从来不是免费的。每个 worker 都消耗 tokens、tool calls、memory、logs、review time，有时还消耗 human attention。如果两个 worker 带着不同假设检查同一段代码，orchestrator 现在就多了一个 conflict-resolution 问题。如果某个 worker 拿到了过宽的 tool sandbox，orchestrator 就要承担更大的 blast radius。

比较安全的默认原则是：当任务有**可分离的不确定性**时，再 spawn sub-agents。比如：

- 主 agent 规划实现时，让 read-only worker 探索代码库。
- 多个独立 benchmark run 使用同一个 return schema。
- Source text 稳定后，委托 translation 和 terminology review。
- Security review 只给只读权限。
- 一个 worker 做慢速外部调研，另一个检查本地代码。

不适合的情况包括：所有步骤都依赖同一个微妙演进中的上下文；merge 比工作本身更难；或者一次高质量 model call 就能低成本解决问题。

---

## 6. Implementation Sketch：一个很小的 Orchestrator

#### 直觉：拿着信封的调度员

把每个 subtask 想成一个信封。调度员写清地址，只放入相关材料，在信封外标出允许使用的工具，并要求按指定格式回信。信封里不应该塞进整个办公室。

下面的代码故意保持很小。它不调用真实 LLM，只展示生产系统会围绕 model call 搭起来的控制结构。

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ToolAccess(Enum):
    READ_ONLY = "read_only"
    SHELL = "shell"
    WRITE = "write"


@dataclass
class SubTask:
    name: str
    objective: str
    context: list[str]
    tools: set[ToolAccess]
    return_schema: str
    timeout_seconds: int = 900


@dataclass
class SubResult:
    name: str
    summary: str
    evidence: list[str]
    confidence: float


def run_subagent(task: SubTask, model_call: Callable[[str], str]) -> SubResult:
    """Build the local prompt and parse a structured result."""
    prompt = f"""
You are a bounded sub-agent.

Objective:
{task.objective}

Allowed tools:
{sorted(t.value for t in task.tools)}

Relevant context:
{chr(10).join('- ' + item for item in task.context)}

Return exactly:
{task.return_schema}
"""
    raw = model_call(prompt)
    return SubResult(
        name=task.name,
        summary=raw[:500],
        evidence=[],
        confidence=0.7,
    )


def orchestrate(tasks: list[SubTask], model_call: Callable[[str], str]) -> str:
    results = [run_subagent(task, model_call) for task in tasks]

    low_confidence = [r.name for r in results if r.confidence < 0.6]
    if low_confidence:
        return f"Need verification before final answer: {low_confidence}"

    merged = "\n\n".join(
        f"## {r.name}\n{r.summary}" for r in results
    )
    return f"Final synthesis:\n\n{merged}"
```

真实 orchestrator 会加入 concurrency、retries、budgets、cancellation、tool mediation、trace storage 和 human approval gates。但核心形状已经在这里：orchestrator 创建 contracts，运行 workers，检查 confidence，并做 synthesis。

---

## 7. Frontier：从手写 Delegation 到 Learnable Orchestration

#### 直觉：从红绿灯到城市交通控制

固定红绿灯在简单路口很有用。城市级交通控制系统则会看拥堵、事故、活动、天气和道路封闭。Agent orchestration 也在沿着类似方向演进：从固定 role prompts，走向能动态决定什么时候 spawn、交给谁、共享什么、何时停止的 adaptive policies。

![图 5：Orchestration frontier timeline](./images/day60/orchestration-frontier-timeline.png)
*图 5：2026 年的新进展正在把 orchestration 从固定角色推向 dynamic specialization、trace-based learning 和 always-on external work queues。*

近期前沿项目：

| 日期 | 项目 | 变化 |
|---|---|---|
| 2026-02-03 / 2026-02-07 | [AOrchestra](https://arxiv.org/abs/2602.03786) | 把 sub-agent 看成 instruction、context、tools、model 的 on-demand composition |
| 2026-05-04 | [Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces](https://arxiv.org/abs/2605.02801) | 把 spawning、delegation、communication、aggregation、stopping 建模为 temporal traces 中可学习的事件 |
| 2026-06-10 | [Orchestra-o1](https://arxiv.org/abs/2606.13707) | 把 orchestration 扩展到 omnimodal tasks，支持 modality-aware decomposition 和 online sub-agent specialization |
| 2026-06 | [OpenAI Symphony](https://openai.com/index/open-source-codex-orchestration-symphony/) | 把项目管理看板变成 always-on control plane，让每个 open task 对应 agent workspace |

2026 年 5 月的 orchestration-traces 论文尤其重要，因为它指出了隐藏的学习问题。Multi-agent system 不只是选择 tokens，还在选择**什么时候 spawn**、**委托给谁**、**如何通信**、**如何聚合**、**什么时候停止**。作者指出，在截至 2026 年 5 月 4 日的 curated pool 里，针对 stopping decision 的显式 reinforcement-learning 方法仍然很少。这个缺口和实践经验一致：启动 agents 很容易，判断“已经够了”更难。

Symphony 展示的是同一趋势在产品侧的形态。它不是让人类手动 juggling 很多 Codex sessions，而是把 task tracker 变成 control plane。OpenAI 的文章说，Symphony 会把 open tasks 映射到 dedicated agent workspaces，并帮助一些团队把 landed pull requests 提高了 500%。这个数字不等于可以泛化到所有团队，但它说明了架构方向：从 interactive sessions，走向 durable work queues、traces、restarts 和 review loops。

---

## 8. 常见误解

### “Agent 越多，答案越好”

不一定。任务能清晰拆分、merge 可控时，更多 agents 才有帮助。如果 workers 重复劳动、继承糟糕 instructions，或者输出无法验证，更多 agents 只会放大问题。

### “Sub-agent 就是 prompt template”

Prompt 只是一部分。真正的 sub-agent 还包括 scoped context、tool permissions、model choice、timeout、sandbox、return schema 和 trace identity。没有这些控制，它只是另一个 chat turn。

### “为了安全，orchestrator 应该把所有 context 都传过去”

通常相反。完整上下文可能泄漏隐私、分散 worker 注意力、掩盖真正目标。更好的模式是最小 briefing，加上 worker 可以检查的 evidence references。

### “Sub-agent orchestration 只适合 coding”

Coding agents 让这个模式更显眼，因为代码库天然有 tools、tests 和 review loops。但同样模式也适用于 customer support、research workflows、compliance review、scientific literature triage、enterprise document processing 和 multimodal analysis。

---

## 9. 延伸阅读

### 入门

1. [OpenAI Codex Subagents](https://developers.openai.com/codex/subagents)  
   了解 Codex 里 subagents 的实践用法，以及用户侧如何接触 orchestration。

2. [Spring AI Agentic Patterns: Subagent Orchestration](https://spring.io/blog/2026/01/27/spring-ai-agentic-patterns-4-task-subagents/)  
   清楚解释 hierarchical subagents、isolated context windows、tool access 和 concurrent execution。

3. [Google Agent Development Kit](https://adk.dev/)  
   从 framework 角度理解 agents、tools、workflow agents 和 multi-agent orchestration。

### 进阶

1. [The Orchestration of Multi-Agent Systems](https://arxiv.org/abs/2601.13671)  
   2026 年 1 月的架构综述，覆盖 orchestration layers、protocols、governance 和 observability。

2. [AdaptOrch](https://arxiv.org/abs/2602.16873)  
   2026 年 2 月论文，主张 topology selection 是一等 optimization target。

3. [AOrchestra](https://arxiv.org/abs/2602.03786)  
   2026 年 2 月论文，用 Instruction-Context-Tools-Model tuple 做动态 sub-agent creation。

4. [Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces](https://arxiv.org/abs/2605.02801)  
   2026 年 5 月论文，把 orchestration events 本身作为 learning object。

5. [Orchestra-o1](https://arxiv.org/abs/2606.13707)  
   2026 年 6 月论文，讨论 omnimodal agent orchestration 和 modality-aware sub-agent specialization。

---

## 思考题

1. 在一个长 coding task 里，哪些内容应该留在主 agent context，哪些更适合委托给 read-only sub-agent？
2. 什么 return schema 会让 reviewer sub-agent 更有用：自由段落、评分表，还是带文件引用的 patch list？
3. Sub-agent 的 isolated context 什么时候保护质量？什么时候会让 worker 漏掉重要全局约束？
4. 如果 orchestration traces 变成训练数据，它们应该受到哪些 privacy 和 governance 规则约束？

---

## 总结

| 概念 | 一句话解释 |
|---|---|
| Orchestrator | 负责分解任务、委托子任务、合并结果并决定何时停止的 agent 或 runtime |
| Sub-agent | 有自己 objective、context、tools、model 和 return contract 的 bounded execution context |
| Topology | 协调形状：parallel、sequential、hierarchical 或 hybrid |
| Minimal contract | 告诉 sub-agent 做什么、读什么、能用什么工具、返回什么格式的 scoped briefing |
| Coordination tax | 使用多个 agents 带来的额外成本、延迟、merge risk 和 safety risk |
| Orchestration trace | 记录 spawning、delegation、communication、tool use、aggregation 和 stopping decisions 的轨迹 |

**Key Takeaway**：Sub-agent orchestration 不是让 agent swarm 变得更吵，而是让长任务变得可治理。Orchestrator 必须只在任务结构值得拆分时才拆，传递最小但足够的上下文，按角色限制 tools，显式合并 evidence，并记录可审计的 traces。整门课最终在这里汇合：model capability 很重要，但可靠的 AI work 依赖模型外面的系统，把能力转化成受控行动。

---

*Day 60 of 60 | LLM Fundamentals*  
*字数：约 4,900 中文字符 | 阅读时间：约 15 分钟*
