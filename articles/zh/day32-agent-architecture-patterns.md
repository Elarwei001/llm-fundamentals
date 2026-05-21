# Day 32: Agent 架构模式 — ReAct、Plan-and-Execute 与更多

> **核心问题**：AI Agent 到底是怎么"思考"和"行动"的？有哪几种基本的架构模式？你该在什么时候用哪一种？

---

## 开篇

想象你是一个侦探在破案。你可以先在犯罪现场深入思考，然后采取一个果断的行动。也可以一边调查一边思考，每发现一个新线索就调整你的理论。两种策略都行——但适用于不同类型的案件。

AI Agent 面临同样的选择。*架构模式*——即 Agent 如何组织推理、行动和观察的循环——是你构建智能体系统时最重要的设计决策。选错了模式，你的 Agent 要么在循环中浪费大量 token，要么无法应对本该处理的复杂情况。

今天，我们拆解四种经典的 Agent 架构模式：**ReAct**、**Plan-and-Execute**、**Reflexion** 和**完全自主循环（Full Autonomous Loop）**。读完之后，你不仅知道每种模式做什么，更知道它*为什么*有效、*什么时候*该用、以及*现实中*各框架如何实现它们。

---

## 1. Agent 循环：一个通用原语

在对比各模式之前，先搞清楚它们的共同点。

#### 直觉：做饭的比喻

每种 Agent 模式都像按菜谱做饭，但"死板程度"不同：

- **ReAct** 就像"边尝边做"：尝一口，调调味，再尝一口。
- **Plan-and-Execute** 就像"周日备餐"：先把一周的菜单全定好，再批量执行。
- **Reflexion** 就像"饭后写点评"：反思哪里做得不好，下次做得更好。
- **完全自主循环** 就像"经营一家餐厅厨房"：计划、烹饪、品尝、调整、再计划——一整天不停循环。

本质上，每个 Agent 都遵循一个循环：

$$
\begin{aligned}
\text{观察} &\rightarrow \text{思考} \rightarrow \text{行动} \rightarrow \text{观察} \rightarrow \cdots
\end{aligned}
$$

区别在于每一步*怎么*组织、规划在*什么时候*发生、以及 Agent *是否*能反思自己的失败。

![四种核心 Agent 架构模式对比](../zh/images/day32/architecture-patterns-comparison-v2.png)
*图 1：四种经典 Agent 架构模式。每种对"观察-思考-行动"循环的组织方式不同。*

---

## 2. 模式一：ReAct — 推理与行动交替

### 核心思想

ReAct（**Re**asoning + **Act**ing 的缩写），由 Yao 等人于 2023 年提出，将"思考"（内部推理）和"行动"（工具调用）交替进行。Agent 不会先把所有事情想清楚再行动——它想一步，做一步，观察结果，再想下一步。

原始论文：["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629)（Yao et al., ICLR 2023）。

#### 直觉：城市探索者

把 ReAct 想象成没有地图时探索一座新城市。你走一个街区，看看周围，判断哪个方向有戏，再走一个街区，如此反复。你不会提前规划完整路线——你边走边调整。

### 运作方式

一个 ReAct 执行轨迹如下：

1. **Thought**："我需要找到法国的首都。"
2. **Action**：`Search["法国首都"]`
3. **Observation**："巴黎是法国的首都。"
4. **Thought**："现在需要巴黎的人口。"
5. **Action**：`Search["巴黎人口"]`
6. **Observation**："巴黎约有 210 万居民。"
7. **Thought**："信息够了。"
8. **Answer**："法国首都巴黎约有 210 万居民。"

![ReAct 执行轨迹示例](../zh/images/day32/react-trace-example-v2.png)
*图 2：一个完整的 ReAct 轨迹。注意推理和行动交替进行——Agent 每次只提前想一步。*

> **深入理解：Thought 是怎么产生的？**
>
> 在上面的执行轨迹中，Thought 2 和 Thought 3 并非预先规划好的——它们是 LLM 在看到上一步的 Observation 之后**实时生成**的。
>
> 每一轮循环中，LLM 看到之前所有的历史（Question + 之前的 Thought/Action/Observation），然后基于这些上下文生成一个新的 Thought + Action。工具执行后返回 Observation，Observation 被追加到历史里，进入下一轮。
>
> 这意味着：
> - LLM 每次只提前想**一步**，不会预想后续所有 Thought
> - 每个 Thought 都是对**最新 Observation 的反应**（这也是 ReAct 名字的由来——Reason + Act 交织）
> - 如果 Observation 2 返回的是"巴黎人口数据不可用"，Thought 3 就会变成"换个数据源再查"，而不是"信息够了"
>
> 这正是 ReAct 灵活性的来源——每一步的思考都取决于上一步实际看到了什么，而不是一开始就锁定的计划。与 Plan-and-Execute 对比就很明显：后者先列出全部步骤，然后机械执行。

### 优缺点

| 方面 | ReAct |
|------|-------|
| **灵活性** | 极佳——能适应意外观察结果 |
| **Token 成本** | 高——每步都需要一次 LLM 调用 |
| **规划深度** | 浅——每次只提前想一步 |
| **最适合** | 开放式任务、探索、调试 |
| **最不适合** | 步骤已知的结构化多步流程 |

### 伪代码

```python
def react_agent(question, tools, llm, max_steps=10):
    context = f"Question: {question}\n"
    for step in range(max_steps):
        # LLM 生成 Thought + Action
        response = llm.generate(context)
        thought, action = parse_thought_action(response)
        context += f"Thought: {thought}\nAction: {action}\n"
        
        if action.type == "Finish":
            return action.answer
        
        # 执行工具并观察
        observation = tools.execute(action)
        context += f"Observation: {observation}\n"
    
    return "超过最大步数限制，未能完成。"
```

---

## 3. 模式二：Plan-and-Execute — 先规划再执行

### 核心思想

Plan-and-Execute 把 Agent 分成两个明确阶段：*规划器*将任务分解为步骤序列，*执行器*逐个运行每个步骤。规划器只运行一次（或偶尔重新规划），执行器负责具体操作。

这一模式通过 ["Plan-and-Solve Prompting"](https://arxiv.org/abs/2305.04091) 论文（Wang et al., 2023）受到广泛关注，并成为 LangGraph 等框架的标准模式。

#### 直觉：项目经理

Plan-and-Execute 就像一位项目经理在周一制定详细的项目计划，然后逐个把任务分配给工程师。PM 不会微观管理每一次按键——他们定好结构，执行交给团队。

### 运作方式

1. **规划**：LLM 将目标分解为有序子任务。
2. **执行**：对每个子任务，Agent 运行相应工具或 LLM 调用。
3. *（可选）重新规划*：如果某步失败或出现新信息，重新调用规划器。

```
任务："比较 3 个供应商的 GPU 价格，推荐最佳方案"
  ↓
规划：
  步骤 1：在 NVIDIA 官店搜索 RTX 5080 价格
  步骤 2：在 Amazon 搜索 RTX 5080 价格
  步骤 3：在新蛋搜索 RTX 5080 价格
  步骤 4：比较所有价格并推荐
  ↓
执行步骤 1 → 执行步骤 2 → 执行步骤 3 → 执行步骤 4 → 回答
```

### 优缺点

| 方面 | Plan-and-Execute |
|------|-----------------|
| **效率** | 好——比 ReAct 更少的 LLM 调用（每任务 3-4 次对 5-7 次） |
| **可预测性** | 高——执行前就能看到计划 |
| **灵活性** | 较低——除非触发重新规划，否则锁定在初始计划中 |
| **最适合** | 步骤可预知的结构化任务 |
| **最不适合** | 高度不可预测的环境，计划很快就会过时 |

### 为什么能节省 Token

处理客户咨询时，ReAct Agent 可能每次交互需要 5-7 次 LLM 调用。Plan-and-Execute Agent 通常只需 3-4 次——一次规划，然后执行。规划调用单次更贵，但总调用次数显著降低。

### 重新规划的触发条件

Plan-and-Execute 中一个关键设计选择是*何时重新规划*。三种常见策略：

1. **从不重新规划**——最简单、最便宜，但任何步骤失败就很脆弱
2. **失败时重新规划**——如果某步返回错误或不可能的结果，把部分结果作为上下文重新调用规划器
3. **定期重新规划**——每 N 步后重新评估剩余计划

策略 2 是生产系统中最常见的。这也是为什么在更高复杂度下，Plan-and-Execute 往往和完全自主循环模式趋同——当重新规划变得频繁时，两者之间的界限就模糊了。

---

## 4. 模式三：Reflexion — 从失败中学习

### 核心思想

Reflexion，由 Shinn 等人提出（NeurIPS 2023），在任何基础 Agent（通常是 ReAct）之上添加了*自我反思*层。Agent 失败时不会直接重试——它会生成一段文字解释*哪里出了问题*，存入记忆，下次避免同样的错误。

论文：["Reflexion: Language Agents with Verbal Reinforcement Learning"](https://arxiv.org/abs/2303.11366)（Shinn et al., NeurIPS 2023）。

#### 直觉：写错题本的学生

想象一个学生每次考试失败后，都会写下自己到底误解了什么。下次考试前，他会复习错题本。Reflexion 就是这样工作的——叫做"语言强化学习"，因为 Agent 用*文字*来强化自己，而不是更新模型权重。

### 运作方式

1. **尝试**：Agent 用 ReAct（或任何基础模式）尝试任务。
2. **评估**：评估器（可以是 LLM 本身）给结果打分。
3. **反思**：如果分数低，Agent 生成自我反思："我失败了，因为我搜索了错误的实体。下次应该先验证实体类型。"
4. **存储**：反思被添加到 Agent 的情景记忆中。
5. **重试**：Agent 再次尝试，现在带着之前的反思作为额外上下文。

$$
\begin{aligned}
\text{Reflection}_k &= \text{LLM}(\text{trace}_k, \text{score}_k, \text{prompt}_{\text{reflect}}) \\
\text{Context}_{k+1} &= \text{Context}_0 \cup \{\text{Reflection}_1, \ldots, \text{Reflection}_k\}
\end{aligned}
$$

### 什么时候有用（什么时候没用）

Reflexion 在以下情况表现出色：
- 任务有明确的成功/失败信号（代码执行、游戏分数）
- 失败是*有信息量的*——理解为什么失败有助于下次改进
- 你可以承担多次尝试的成本

Reflexion 在以下情况下表现不佳：
- 任务太难，反思之后 Agent 每次*犯不同的错误*
- 没有明确的评估信号
- 单次尝试场景（生产数据库迁移不能重试）

### "语言强化学习"的深层含义

Reflexion 被称为"语言强化学习"，因为它镜像了传统 RL——但不是通过梯度下降更新模型权重，Agent 更新的是基于文本的记忆缓冲区。"奖励信号"是评估分数，"策略更新"是自我反思文本。这里有一个深刻的洞察：*语言本身成为学习的媒介*，而非数值参数。代价是基于文本的更新比梯度更新更嘈杂、更不系统——但它们可解释、可组合，而且不需要模型重新训练。

---

## 5. 模式四：完全自主循环

### 核心思想

完全自主循环将规划、执行*和*反思组合成持续循环。与 Plan-and-Execute（规划一次）不同，这个模式在每轮执行后根据新观察重新规划。与 Reflexion（只在失败时反思）不同，这个模式持续反思。

#### 直觉：创业公司 CEO

创业公司 CEO 不会在 1 月 1 日规划全年。他们先规划季度，执行一个月，复盘数据，调整计划，继续执行，如此反复。完全自主循环也是这样——持续的计划→执行→反思→再计划循环。

### 运作方式

```
目标："撰写 AI 编程工具市场的竞品分析报告"
  ↓
[规划] 分解为：调研市场 → 识别工具 → 分析功能 → 撰写报告
  ↓
[执行] 调研市场 → 观察发现
  ↓
[反思] "市场正在转向智能体工具。我应该调整分析框架。"
  ↓
[重新规划] 增加子任务：分析智能体编程能力
  ↓
[执行] 继续调整后的计划...
  ↓
[反思] "报告太长了。只聚焦前 5 个工具。"
  ↓
[重新规划] 缩小范围...
  ↓
[完成] 交付最终报告
```

### 风险：无限循环

完全自主循环最大的风险是 Agent 陷入死循环——永远在重新规划、原地打转、或钻进越来越偏门的子任务。应对措施包括：

- **最大迭代限制**（硬性上限循环次数）
- **进度检查**（每次迭代要求可衡量的进展）
- **人类介入检查点**（每 N 步暂停等待确认）
- **预算约束**（token 成本超过阈值时停止）

---

## 6. 模式对比

| 模式 | 规划方式 | 执行方式 | 反思 | Token 成本 | 可靠性 |
|------|---------|---------|------|-----------|--------|
| **ReAct** | 隐式，单步 | 交替进行 | 无 | 高 | 中等 |
| **Plan-and-Execute** | 显式，预先 | 顺序执行 | 无 | 中等 | 高 |
| **Reflexion** | 隐式（基础 Agent） | 交替进行 | 失败时 | 很高 | 中高 |
| **自主循环** | 持续重新规划 | 迭代执行 | 持续 | 最高 | 不确定 |

### 核心权衡：灵活性 vs. 效率

- **更灵活**（ReAct、自主循环）→ 处理意外更好，但成本更高
- **更结构化**（Plan-and-Execute）→ 高效但计划断裂时脆弱
- **更反思**（Reflexion、自主循环）→ 从失败中学习，但消耗更多 token

![Agent 模式选择决策树](../zh/images/day32/pattern-decision-tree-v2.png)
*图 3：根据任务特征选择正确 Agent 架构模式的决策指南。*

---

## 7. 真实框架实现

### LangGraph（LangChain）

[LangGraph](https://github.com/langchain-ai/langgraph) 于 2025 年 10 月发布 v1.0，是目前采用最广泛的 Agent 编排框架。它将所有四种模式实现为基于图的工作流，核心抽象是*状态图*——节点是 Agent 步骤，边是状态转换。

LangGraph 提供预置模式：
- `create_react_agent` — 即用型 ReAct 循环
- Plan-and-Execute 图 — 规划器节点 + 执行器节点
- Reflexion — 内置评估和重试循环

### OpenAI Agents SDK

[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) 于 2026 年 3 月发布，2026 年 4 月更新至 v2（新增原生沙箱执行），采用极简原语方法。仅提供三个原语：Agent、Handoff 和 Guardrail。默认使用 ReAct 循环；更复杂的架构通过 Agent 交接组合。

### Google ADK

[Google Agent Development Kit (ADK)](https://github.com/google/adk-python) 于 2026 年 4 月推出，与 Google Cloud 和 Gemini 原生集成。默认使用 Plan-and-Execute 模式，体现了 Google 对结构化企业工作流的侧重。

### Anthropic 的方案

Anthropic 的 [Claude Computer Use](https://www.anthropic.com/news/claude-computer-use)（2026 年 3 月 GA）实现了完全自主循环模式——Claude 直接控制电脑，点击、打字、导航应用程序。这是目前生产中最激进的 Agent 自主形式。

---

## 8. 前沿进展（2025–2026）

Agent 架构领域正在快速演进：

1. **OpenAI Agents SDK v2（2026 年 4 月）**——新增原生沙箱执行和模型原生 harness，让长时间运行的自主 Agent 在企业场景更安全。([OpenAI 博客](https://openai.com/index/the-next-evolution-of-the-agents-sdk/), 2026 年 4 月)

2. **Google ADK（2026 年 4 月）**——Google 对 Agent SDK 竞赛的回应，与 Gemini 和 Google Cloud 深度集成，开箱支持多 Agent 编排。

3. **LangGraph 达到生产稳定（2026 年 Q1）**——现已在生产环境运行，支持内置检查点和子图组合。([LangChain 博客](https://blog.langchain.com/langchain-langgraph-1dot0/), 2025 年 10 月)

4. **Anthropic Claude Computer Use GA（2026 年 3 月）**——Claude 现在可以自主控制桌面应用，代表了目前最大胆的自主循环部署。([CNBC 报道](https://www.cnbc.com/2026/03/24/anthropic-claude-ai-agent-use-computer-finish-tasks.html), 2026 年 3 月)

5. **ReAcTree（2025 年 11 月）**——ReAct 的层级变体，将 Agent 推理组织为带控制流的树结构，在长周期任务上表现更好。([arXiv:2511.02424](https://arxiv.org/abs/2511.02424))

6. **"Agentic AI: Architectures, Taxonomies, and Evaluation" 综述（2026 年 1 月）**——覆盖单 Agent 和多 Agent 模式的综合分类法，正式化了这一领域。([arXiv:2601.01743](https://arxiv.org/abs/2601.01743))

![Agent 框架时间线](../zh/images/day32/framework-timeline-v2.png)
*图 4：2023 年至 2026 年中的关键 Agent 框架和论文时间线。*

---

## 9. 常见误解

### ❌ "ReAct 总是比 Plan-and-Execute 好"

不是。ReAct 的灵活性有代价——更多 LLM 调用、更高延迟、更不可预测的行为。对于结构化任务（例如"从这 3 个 API 拉取数据，合并并总结"），Plan-and-Execute 既更便宜又更可靠。

### ❌ "自主 Agent 可以永远运行不需要监督"

理论上可以。实践中，自主 Agent 会漂移、幻觉出新的目标、陷入无意义的循环。每个生产环境的自主 Agent 都需要护栏：迭代限制、预算上限、人工检查点。

### ❌ "你必须为整个系统选择一种模式"

真实系统经常*组合*多种模式。一种常见架构是：外层用 Plan-and-Execute，各步骤内部用 ReAct 执行，关键步骤加上 Reflexion 确保准确性。

---

## 10. 代码示例：Python 实现 ReAct Agent

```python
import json
from typing import List, Dict, Optional

class Tool:
    """Agent 可以调用的简单工具。"""
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

class ReActAgent:
    """一个最小化的 ReAct Agent，交替推理和行动。"""
    
    SYSTEM_PROMPT = """你是一个 ReAct Agent。每一步：
1. 输出 "Thought: <你的推理>"
2. 输出 "Action: <工具名>(<输入>)" 
3. 等待观察结果
4. 重复，直到能用 "Action: Finish(<答案>)" 回答

可用工具：
{tool_descriptions}"""
    
    def __init__(self, llm, tools: List[Tool], max_iterations: int = 8):
        self.llm = llm  # 任何带有 .generate() 方法的 LLM 客户端
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
    
    def run(self, question: str) -> str:
        tool_descs = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools.values()
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(
                tool_descriptions=tool_descs
            )},
            {"role": "user", "content": question}
        ]
        
        for _ in range(self.max_iterations):
            response = self.llm.generate(messages)
            messages.append({"role": "assistant", "content": response})
            
            # 从响应中解析 Action
            action = self._parse_action(response)
            if action is None:
                continue
            
            if action["tool"] == "Finish":
                return action["input"]
            
            # 执行工具
            if action["tool"] in self.tools:
                observation = self.tools[action["tool"]].func(action["input"])
            else:
                observation = f"错误：未知工具 '{action['tool']}'"
            
            messages.append({
                "role": "user", 
                "content": f"Observation: {observation}"
            })
        
        return "Agent 超过最大迭代次数，未能完成。"
    
    def _parse_action(self, text: str) -> Optional[Dict]:
        """从 LLM 响应中提取 Action。"""
        for line in text.split("\n"):
            if line.startswith("Action:"):
                content = line[len("Action:"):].strip()
                paren_idx = content.find("(")
                if paren_idx > 0:
                    tool = content[:paren_idx]
                    inp = content[paren_idx+1:-1]
                    return {"tool": tool, "input": inp}
        return None


# --- 使用示例 ---
def fake_search(query: str) -> str:
    """用于演示的模拟搜索工具。"""
    db = {
        "法国首都": "巴黎是法国的首都和最大城市。",
        "巴黎人口": "巴黎市区：约 210 万（2025 年）。都会区：约 1240 万。",
    }
    return db.get(query.lower(), f"未找到 '{query}' 的结果。")

tools = [
    Tool("Search", "搜索网络获取信息", fake_search),
]
# agent = ReActAgent(llm=my_llm_client, tools=tools)
# result = agent.run("法国首都有多少人口？")
```

---

## 11. 延伸阅读

### 基础论文

| 论文 | 年份 | 贡献 |
|------|------|------|
| ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629) | 2023 | 提出 ReAct 模式 |
| ["Reflexion: Language Agents with Verbal Reinforcement Learning"](https://arxiv.org/abs/2303.11366) | 2023 | Agent 自我反思机制 |
| ["Plan-and-Solve Prompting"](https://arxiv.org/abs/2305.04091) | 2023 | 结构化规划分解 |
| ["Tree of Thoughts"](https://arxiv.org/abs/2305.10601) | 2023 | 基于搜索的思维树推理 |

### 综述和分类

1. ["Agentic AI: Architectures, Taxonomies, and Evaluation"](https://arxiv.org/abs/2601.01743) — 2026 年全面综述，覆盖单 Agent 和多 Agent 模式
2. ["AI Agent Systems: Architectures, Applications, and Evaluation"](https://arxiv.org/abs/2601.12560) — Agent 组件和编排模式分类法（2026 年 1 月）

### 框架和工具

1. [LangGraph](https://github.com/langchain-ai/langgraph) — 基于图的 Agent 编排（v1.0, 2025 年 10 月）
2. [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 极简原语 Agent 框架（2026 年 3 月）
3. [Google ADK](https://github.com/google/adk-python) — Google 的 Agent 开发工具包（2026 年 4 月）
4. [Google Cloud: 为智能体 AI 系统选择设计模式](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system) — Google 官方架构指南

---

## 思考题

1. 为什么 ReAct 在需要长周期规划的任务上表现不佳？它具体会产生什么样的失败模式？
2. 如果 Reflexion 的自我反思是由犯错的同一个 LLM 生成的，我们凭什么信任反思的质量？可能出现什么问题？
3. 什么时候你会组合多种模式（例如 Plan-and-Execute 外层循环 + ReAct 内部步骤），而不是只用一种模式？权衡是什么？

---

## 总结

| 概念 | 一句话解释 |
|------|-----------|
| ReAct | 推理和行动交替进行；像没有地图的探索，边走边看 |
| Plan-and-Execute | 先分解再执行；像项目经理的甘特图，结构清晰 |
| Reflexion | 失败后自我反思；像学生写错题本，从错误中学习 |
| 完全自主循环 | 持续的规划-执行-反思循环；像创业公司 CEO 按季度复盘 |
| 模式组合 | 真实系统在架构不同层面组合多种模式 |
| 护栏 | 每个自主 Agent 都需要迭代限制、预算上限和人工检查点 |

**核心收获**：没有单一的"最佳"Agent 架构模式。正确的选择取决于任务的结构、你对成本和延迟的容忍度、以及环境的可预测性。大多数生产系统组合多种模式——在可能的地方使用结构化，在必要的地方保留灵活性。从能工作的最简模式开始，只在任务需要时才增加复杂度。

---

*Day 32 of 60 | LLM Fundamentals*
*字数：~3200 | 阅读时间：~16 分钟*
