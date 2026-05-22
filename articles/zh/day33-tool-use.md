# Day 33: 工具使用 — LLM 如何突破训练数据的边界

> **核心问题**: LLM 如何调用外部函数、API 和工具与现实世界交互？为什么这项能力是把聊天机器人变成智能体的关键？

---

## 开篇

想象你是一个才华横溢的图书管理员，背下了世上所有的书。有人问你："东京现在天气怎么样？"你能描述什么是天气、解释气象系统、甚至背诵历史天气规律——但你无法告诉他*此刻*的温度，因为你被锁在一间没有窗户的房间里。

LLM 的处境完全一样。它从训练数据中吸收了海量知识，但它无法查询实时信息、精确计算、访问数据库或在现实中采取行动。工具使用（Tool Use）就是我们为它打开的那扇窗。

2023 年之前，LLM 是令人印象深刻的文本生成器。工具使用成为标配之后，它们变成了*智能体*——能够推理自己需要什么、主动获取、并根据结果行动的系统。这种转变的重要性，可以说超过了同期任何模型规模的提升。

---

## 1. 什么是工具使用？

#### 直觉：厨师与厨房

把 LLM 想象成一个背下了所有菜谱的大厨。大厨可以*描述*如何做完美的舒芙蕾，但要真做出来，他需要工具：烤箱、打蛋器、冰箱里的食材。工具使用就是给这位大厨进厨房的权限。大厨决定用*什么*工具、*怎么*用，但工具做实际的物理工作。

形式化地说，**工具使用**（也叫 **函数调用** 或 **工具调用**）指的是赋予 LLM 以下能力：

1. **判断**何时需要外部帮助（计算、搜索、数据库查询）
2. **指定**调用哪个工具、传入什么参数
3. **接收**工具的输出并整合到回复中
4. **重复**以上过程（多步工具使用）

### 工具使用循环

![图 1：工具使用循环](./images/day33/tool-use-loop.png)
*图 1：核心循环——用户提问触发 LLM 推理，可能产生工具调用。工具执行后返回结果，LLM 再决定是回复还是继续调用工具。*

这个循环是每个 AI 智能体的基本构建单元。在 Day 31 我们看到了 ReAct 模式；今天我们深入让这个模式运转的*底层机制*。

---

## 2. 函数调用：API 机制详解

#### 直觉：餐厅点菜

函数调用就像在餐厅点菜。你（LLM）不亲自做菜。你先看菜单（工具的 Schema），决定要什么，然后用具体的参数下单（"牛排，五分熟，不要酱汁"）。厨房（外部系统）做好后端上来。你再用自己方式呈现给客人。

### 逐步流程

![图 2：函数调用流程](./images/day33/function-calling-flow.png)
*图 2：函数调用的五个步骤——从用户消息到最终 LLM 回复。*

具体发生了什么：

**步骤 1 — 用户发送消息。** "东京天气怎么样？"

**步骤 2 — LLM 接收可用工具。** 连同用户消息一起，开发者提供一组工具定义，每个工具由 JSON Schema 描述：

```json
{
  "name": "get_weather",
  "description": "获取指定城市的当前天气",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "城市名称"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["city"]
  }
}
```

**步骤 3 — LLM 决定调用工具。** 模型不生成文本回复，而是输出一个结构化的工具调用请求：

```json
{
  "name": "get_weather",
  "arguments": {"city": "Tokyo", "unit": "celsius"}
}
```

**步骤 4 — 开发者执行函数。** LLM 不运行代码，是*开发者的代码*去调用真正的天气 API 并返回结果：

```json
{"temperature": 22, "condition": "sunny", "humidity": 65}
```

**步骤 5 — LLM 生成最终回复。** "东京目前气温 22°C，晴天，湿度 65%。"

### 关键设计决策：LLM 不执行

值得强调：LLM **从不自己运行工具**。它输出一个结构化的请求，由外围的应用代码来执行。这是一条关键的安全边界——模型可以*请求*操作，但开发者控制哪些请求真正被执行。

---

## 3. 主要 API 提供商对比

| 特性 | OpenAI | Anthropic | Google Gemini |
|------|--------|-----------|---------------|
| **API 名称** | Function Calling | Tool Use | Function Calling |
| **发布时间** | 2023 年 7 月 | 2024 年 4 月 | 2023 年末 |
| **并行调用** | 支持（默认开启） | 支持 | 支持 |
| **严格模式** | 支持（结构化输出） | 支持（tool_choice） | 支持 |
| **强制使用工具** | `tool_choice: required` | `tool_choice: any` | `function_calling_config` |
| **Schema 格式** | JSON Schema | JSON Schema | JSON Schema |

### OpenAI 函数调用

OpenAI 在 2023 年 7 月率先普及了结构化函数调用。核心特性：

- **并行函数调用**：模型可以在一次回复中调用多个工具（比如同时查询三个城市的天气）
- **严格模式**：`strict: true` 保证模型输出完全符合 JSON Schema——不缺字段、不弄错类型
- **结构化输出**：自 2024 年 8 月起，函数调用基于结构化输出基础设施，确保可靠性

```python
from openai import OpenAI
client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        },
        "strict": True  # 保证 Schema 合规
    }
}]

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "东京天气？"}],
    tools=tools
)

# 检查模型是否要调用工具
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(tool_call.function.name)      # "get_weather"
    print(tool_call.function.arguments)  # '{"city": "Tokyo"}'
```

### Anthropic 工具使用

Anthropic 的方案设计类似，但有一些差异：

- **`tool_choice`**：可以强制使用特定工具（`{"type": "tool", "name": "get_weather"}`）或任意工具（`{"type": "any"}`）
- **丰富的工具结果**：工具返回的结果可以包含图片，不限于文本
- **思维链整合**：在扩展思考模式下，模型会先推理该用哪个工具再调用

### Google Gemini 函数调用

Google 的 Gemini 模型同样支持函数调用：
- **自动函数推荐**：根据对话上下文自动建议使用哪个函数
- **函数调用配置**：精细控制是否允许、禁止或强制函数调用

---

## 4. 模型如何学习工具使用？

#### 直觉：教人使用新库

想象教一个程序员使用新的 API 库。你可以：(a) 把文档甩给他，让他自己摸索（提示方法），(b) 带他看几个例子，让他跟着练习（微调方法），或者 (c) 让他动手尝试，在错误中给反馈，通过试错不断进步（强化学习方法）。

![图 3：三种工具使用训练方法对比](./images/day33/tool-training-approaches.png)
*图 3：提示方法、微调和强化学习在可靠性、灵活性和实现复杂度上各有取舍。*

### 方法 1：上下文学习（提示方法）

最简单的方案：在系统提示中包含工具描述，模型凭预训练知识推断如何使用。

- **优点**：零额外训练，任何模型立即可用
- **缺点**：工具太多时可靠性下降（通常超过 10-20 个就开始退化），消耗上下文 Token，格式合规性不保证
- **适用场景**：原型验证、小工具集、已经针对工具使用微调过的模型

### 方法 2：监督微调

用正确的工具使用示例训练模型——模型在对话中正确判断何时及如何调用工具的数据。

关键工作：**Toolformer**（Meta，2023 年 2 月，[arXiv:2302.04761](https://arxiv.org/abs/2302.04761)）证明了 LLM 可以通过在训练数据中插入 API 调用来自我学习工具使用。模型学会了预测在*哪里*以及使用*哪个*工具。

后来，**TL-Training**（Ye et al.，2024 年 12 月，[arXiv:2412.15495](https://arxiv.org/abs/2412.15495)）提出了一种基于任务特征的框架，通过将工具交互分解为任务特征来改进工具使用训练，在泛化到未见过的工具时表现更好。

- **优点**：格式合规可靠，对新工具泛化更好
- **缺点**：需要精心策划的训练数据，微调的计算成本
- **适用场景**：已知工具集的生产部署

### 方法 3：强化学习

用反馈训练模型：奖励成功的工具使用，惩罚失败。

关键工作：**"From Exploration to Mastery"**（2024 年 10 月，[arXiv:2410.08197](https://arxiv.org/abs/2410.08197)）让 LLM 通过自主交互来掌握工具——模型通过试错探索工具的能力，从执行反馈中学习。

**"Self-Training for Tool-Use Without Demonstrations"**（2025 年 2 月，[arXiv:2502.05867](https://arxiv.org/abs/2502.05867)）证明 LLM 可以通过自我生成的轨迹和执行反馈来学习工具使用，无需任何人类示范。

- **优点**：优雅地处理错误，自我改进，鲁棒性最强
- **缺点**：训练流水线复杂，奖励设计困难
- **适用场景**：需要故障恢复的复杂工具环境

### 现代前沿模型如何做

实际上三种方法会组合使用。GPT-5.5 和 Claude Opus 4.7 这样的前沿模型：

1. 在包含代码、API 文档和结构化数据的大规模数据上**预训练**
2. 在精心策划的工具使用示例上**微调**
3. 通过 **RLHF** 训练优先选择有用的工具使用而非幻觉回复
4. 最新的前沿模型（如 GPT-5.5）还引入了**强化学习**来优化复杂的多步工具编排，并在 agentic 任务上进行了专门的后训练

这种分层方法解释了为什么现代模型能够可靠地调用从未见过的工具——它们内化了"读 Schema → 判断是否调用 → 正确构造参数"这个*模式*。

---

## 5. 工具使用设计模式

### 模式 1：单次工具调用

最简单的模式：用户提问，模型调用一个工具，返回答案。

```
用户: "$85 的小费给 15% 是多少？"
LLM: [调用 calculator(85, 0.15)]
工具: 12.75
LLM: "$85 的 15% 小费是 $12.75。"
```

### 模式 2：并行工具调用

模型同时调用多个独立的工具：

```
用户: "比较纽约、伦敦和东京的天气"
LLM: [并行调用 get_weather("NYC"), get_weather("London"), get_weather("Tokyo")]
工具: {NYC: 18°C, 伦敦: 12°C, 东京: 22°C}
LLM: "东京最暖和，22°C，其次是纽约 18°C，伦敦 12°C。"
```

### 模式 3：链式工具调用

模型依次调用工具，前一个调用的输出作为下一个的输入：

```
用户: "帮我在巴黎酒店附近订一家餐厅"
LLM: [调用 get_hotel_location(user_id)] → "Le Marais, Paris"
LLM: [调用 search_restaurants("Le Marais, Paris")]
LLM: [调用 book_restaurant(最佳结果, user_id)]
```

### 模式 4：错误处理与重试

稳健的工具使用需要处理失败情况：

```
LLM: [调用 get_weather("Tkoyo")] → 错误: 找不到该城市
LLM: "找不到 'Tkoyo'，您是指东京 (Tokyo) 吗？"
LLM: [调用 get_weather("Tokyo")] → 成功
```

这种错误恢复模式是微调和 RL 训练模型显著优于纯提示方法的地方。

---

## 6. LLM 生态中的常见工具

| 工具类别 | 示例 | 功能 |
|---------|------|------|
| **网络搜索** | Brave API、Google Search、Tavily | 获取实时信息 |
| **代码执行** | Python 沙箱、Code Interpreter | 精确计算、数据分析 |
| **文件操作** | 读写文件、搜索文件 | 文档处理、数据提取 |
| **数据库** | SQL 查询、向量搜索 | 结构化数据检索 |
| **通信** | 邮件、Slack、短信 | 在现实中采取行动 |
| **浏览器** | Puppeteer、Playwright | 网页导航、表单填写 |
| **API** | 任意 REST/GraphQL API | 无限扩展能力 |

---

## 7. 模型上下文协议（MCP）：工具访问的标准化

工具使用生态中最重要的发展之一是 **MCP（Model Context Protocol，模型上下文协议）**，由 Anthropic 于 2024 年 11 月推出。MCP 的目标是为 AI 工具做 USB 为外设做的事情——创建一个通用标准。

### MCP 为什么重要

MCP 之前，每个 AI 应用都以不同方式实现工具集成。想让 LLM 访问 GitHub？写一个自定义函数。想用 Google Drive？再写一个。每个集成都是定制开发。

MCP 提供：
- **标准协议**，用 JSON-RPC 2.0 向 LLM 暴露工具
- **可复用的工具服务器**——写一次，任何 MCP 兼容客户端都能用
- **不断增长的生态系统**，预置了流行服务的服务器

### 关键里程碑

![图 4：LLM 工具使用演进时间线](./images/day33/tool-use-timeline.png)
*图 4：从 Toolformer（2023 年 2 月）到 Agentic AI Foundation（2025 年 12 月），工具使用生态快速成熟。*

- **2024 年 11 月**：Anthropic 发布 MCP
- **2025 年 3 月**：OpenAI 采纳 MCP，标志跨行业支持
- **2025 年 11 月**：OpenAI 和 Anthropic 联合发布 **MCP Apps**，为协议添加了 UI 组件（[blog.modelcontextprotocol.io](https://blog.modelcontextprotocol.io/posts/2025-11-21-mcp-apps/)）
- **2025 年 12 月**：Anthropic 将 MCP 捐赠给 Linux 基金会下的 **Agentic AI Foundation (AAIF)**，由 Anthropic、Block 和 OpenAI 联合创立，Google 参与支持
- **2026 年初**：Google 采纳 MCP。截至 2026 年 5 月，MCP 累计安装量达 9700 万，生态中有超过 20 万个 MCP 服务器

### 安全隐忧

快速增长并非没有问题。2026 年 5 月，研究人员披露 20 万个 MCP 服务器存在命令执行漏洞（CVE-2026-30623），凸显了协议宽松默认设置的风险。这提醒我们：标准化工具访问同时放大了能力和风险。

### MCP 与传统集成方式的对比

要理解 MCP 为什么重要，需要看清它和已有的接口标准有什么本质区别：

| 维度 | REST API | gRPC / RPC | MCP |
|------|----------|------------|-----|
| **设计目的** | 为人类开发者设计的通用接口 | 为微服务间高性能调用设计 | **专为 AI 模型消费设计** |
| **发现机制** | 需要人类阅读文档 | 需要人类查阅 .proto 文件 | `tools/list` 动态发现，模型自己能理解 |
| **描述格式** | OpenAPI/Swagger（可选） | Protocol Buffers（强类型） | JSON Schema + 自然语言 description |
| **协议** | HTTP + JSON/XML | HTTP/2 + Protobuf | JSON-RPC 2.0（stdio / SSE / Streamable HTTP） |
| **谁在"读"接口** | 人类开发者 | 人类开发者 | **AI 模型** |
| **工具粒度** | 粗粒度（资源 + CRUD） | 细粒度（方法调用） | 中粒度（按任务能力组织） |

核心区别在于表格中"谁在'读'接口"这一行——**MCP 是第一个"为模型读"而设计的协议**。REST 和 RPC 的消费者是人类程序员：人读文档、写胶水代码、调试接口。MCP 的消费者是 AI 模型：模型通过 `tools/list` 自主发现可用工具，通过 JSON Schema 理解参数结构，通过 description 字段的自然语言理解工具语义，然后直接构造调用。

#### 一个具体例子

假设你要让 LLM 访问用户的 Google Drive 文件：

**传统 REST API 方式**：开发者查阅 Google Drive API 文档 → 写认证逻辑 → 为每个操作写函数包装 → 把这些函数注册为 LLM 工具 → 维护两边的一致性

**MCP 方式**：启动一个 Google Drive MCP Server → LLM 通过 `tools/list` 发现 `list_files`、`read_file`、`search_files` 等工具 → 直接调用。开发者不需要写包装代码。

此外，MCP 还提供了一些传统 API 没有的能力：
- **Prompts（提示模板）**：MCP Server 可以暴露预写的提示模板，告诉模型如何更好地使用工具
- **Resources（资源）**：Server 可以提供上下文数据（如文件内容、数据库 schema），而不仅是可调用的函数
- **Sampling（采样请求）**：Server 可以反向请求 LLM 完成子任务，实现 Server ↔ LLM 的双向协作

这些能力让 MCP 不只是一个"远程过程调用"协议，而是一个 **AI 与外部世界的完整交互框架**。

---

## 8. 前沿进展

### 推理模型与工具使用

最新前沿模型（OpenAI o 系列、带扩展思考的 Claude、DeepSeek-R1）将思维链推理与工具使用结合。模型不再立即调用工具，而是先*思考*是否需要工具、该传什么参数、拿到结果后怎么做。这大大提高了复杂多步任务的准确率。

### 大规模工具学习

**ToolBench**（Qin et al.，2024，[arXiv:2307.16789](https://arxiv.org/abs/2307.16789)）来自 Gorilla 项目，创建了一个包含 16,464 个真实世界 API、覆盖 49 个类别的基准测试，能够系统评估工具使用能力。

---

## 9. 代码示例：构建一个使用工具的智能体

```python
import json

# 定义可用工具
tools = [
    {
        "name": "calculator",
        "description": "计算数学表达式",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，如 '2 + 3 * 4'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_stock_price",
        "description": "获取当前股票价格",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "股票代码，如 'AAPL'"
                }
            },
            "required": ["ticker"]
        }
    }
]

# 工具实现
def execute_tool(name, args):
    """执行工具调用并返回结果。"""
    if name == "calculator":
        # 安全的数学表达式求值
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in args["expression"]):
            return str(eval(args["expression"]))
        return "Error: invalid expression"
    elif name == "get_stock_price":
        # 模拟股票价格查询
        prices = {"AAPL": 198.50, "GOOGL": 175.30, "TSLA": 245.60}
        ticker = args["ticker"].upper()
        if ticker in prices:
            return json.dumps({"ticker": ticker, "price": prices[ticker]})
        return json.dumps({"error": f"Unknown ticker: {ticker}"})
    return json.dumps({"error": f"Unknown tool: {name}"})

# 智能体循环
def agent_loop(client, user_message, tools, max_turns=5):
    """运行工具使用循环，直到模型给出最终回答。"""
    messages = [{"role": "user", "content": user_message}]
    
    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=[{"type": "function", "function": t} for t in tools]
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        # 如果没有工具调用，说明得到了最终回答
        if not msg.tool_calls:
            return msg.content
        
        # 执行每个工具调用并添加结果
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = execute_tool(name, args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    
    return "智能体超过了最大工具使用轮次。"

# 使用示例（伪代码，需要 OpenAI 客户端）：
# answer = agent_loop(client,
#     "如果我买 10 股 AAPL 和 5 股 TSLA，总共多少钱？",
#     tools)
# 模型会：
# 1. 调用 get_stock_price("AAPL") → 198.50
# 2. 调用 get_stock_price("TSLA") → 245.60
# 3. 调用 calculator("10 * 198.50 + 5 * 245.60") → 3213
# 4. 返回："总共需要 $3,213。"
```

---

## 10. 常见误解

### 误解："LLM 自己执行工具"

模型只是用结构化 JSON *建议*一次工具调用。你的应用代码必须执行它并把结果喂回去。模型从不直接运行代码、发起 HTTP 请求或访问数据库。这是出于安全和控制的设计选择。

### 误解："工具越多，智能体越强"

工具太多反而降低性能。研究表明，当模型需要从 20 个以上工具中选择时，准确率会下降，因为选择本身变得更难。最佳实践：每个智能体保持 5-15 个工具，或使用两级方法，先用路由模型选出相关子集。

### 误解："工具使用就是提示工程"

虽然提示可以让有能力的模型实现基本的工具使用，但生产级的工具使用需要：(a) 健壮的 Schema 验证，(b) 错误处理和重试逻辑，(c) 速率限制和超时管理，(d) 安全沙箱。提示只是更大系统中的一个组件。

### 误解："函数调用和 RAG 是一回事"

RAG（检索增强生成，Day 35 详解）检索相关文档来增强模型的上下文。函数调用让模型*采取行动*。两者互补：一次函数调用可能触发 RAG 检索，但它们的用途不同。

---

## 11. 延伸阅读

### 基础论文

1. ["Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761)（Meta，2023 年 2 月）——证明 LLM 可以通过在训练数据中插入 API 调用来自学工具使用
2. ["Gorilla: Large Language Model Connected with Massive APIs"](https://arxiv.org/abs/2305.15334)（UC Berkeley，2023 年 5 月）——展示了大规模微调工具使用，涵盖 1,645 个 API
3. ["ToolBench: Facilitating Large Language Models to Master 16000+ Real-world APIs"](https://arxiv.org/abs/2307.16789)（Qin et al.，2024）——大规模工具使用能力评估基准

### 最新研究

4. ["TL-Training: A Task-Feature-Based Framework for Training LLMs in Tool Use"](https://arxiv.org/abs/2412.15495)（Ye et al.，2024 年 12 月）——通过任务特征分解提升泛化能力
5. ["From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions"](https://arxiv.org/abs/2410.08197)（2024 年 10 月）——通过自主交互学习工具
6. ["Self-Training LLMs for Tool-Use Without Demonstrations"](https://arxiv.org/abs/2502.05867)（2025 年 2 月）——无需人类示范的自我训练

### 实践资源

7. [OpenAI 函数调用指南](https://developers.openai.com/api/docs/guides/function-calling)——OpenAI 官方函数调用文档
8. [Anthropic 工具使用文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)——Claude 工具使用功能官方指南
9. [Model Context Protocol 规范](https://modelcontextprotocol.io/)——连接 AI 模型与工具和数据源的开放标准

---

## 思考题

1. 如果 LLM 可以调用*任何*工具，是什么阻止它调用危险工具？谁应该负责——模型、开发者还是用户？
2. 为什么顺序工具调用对当前 LLM 来说比并行调用更难？链式调用需要什么样的推理能力？
3. MCP 标准化了工具访问，但标准化是否可能造成单一文化——一个漏洞就影响所有系统？

---

## 总结

| 概念 | 一句话解释 |
|------|-----------|
| 工具使用 | LLM 调用外部函数来获取文本生成之外的能力 |
| 函数调用 | LLM 输出结构化工具请求（JSON）供开发者代码执行的 API 机制 |
| 工具 Schema | 定义工具名称、描述和预期参数的 JSON Schema |
| 并行调用 | 在一次模型回复中发起多个独立的工具调用 |
| 链式调用 | 顺序的工具调用，前一个结果作为下一个的输入 |
| MCP | 模型上下文协议——连接 LLM 与工具的开放标准（基于 JSON-RPC） |
| Toolformer | Meta 2023 年的论文，证明 LLM 能从 API 增强数据中自学工具使用 |

**核心要点**：工具使用是语言理解通向现实世界行动的桥梁。LLM 决定*做什么*，但从不亲自执行——这种分离既是安全特性也是设计原则。随着 MCP 标准化工具的描述和连接方式，我们正在从定制集成走向 AI 智能体的通用工具生态。

---

*Day 33 of 60 | LLM Fundamentals*
*字数：约 3200 | 阅读时间：约 15 分钟*
