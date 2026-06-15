# Day 45: Agent-Native Workflows — 从散乱工具到编排化 Skill Pipeline

> **核心问题**：当团队有 20 个 skill 和 10 种日常流程时，怎么把它们变成一个可靠的、agent 可以按需调用的 workflow 系统？

---

## 开篇：工具间的困境

每个木工车间都有挂板。锤子挂这里，锯子放那里，卷尺在抽屉里。当你清楚每件工具的位置、知道该拿哪一把时，工作行云流水。但当工具堆积如山——重复的、坏掉的混在一起、没有任何标记——你花在找工具上的时间比干活还多。

AI agent 团队很快就会撞上这堵墙。在 [Day 40](day40-agent-skills.md) 里，我们学了怎么设计单个 skill：用 instructions、knowledge 和 constraints 把一个工具包起来。5 个 skill 的时候这很好用。到 20 个呢？完全不同的问题。Skill 之间有重叠。Agent 选错了。没人记得哪个 skill 处理哪个 edge case。而"让 agent 自己看着办"——把所有 skill description 塞进 context window 然后指望 agent 选对——在生产环境里开始撑不住了。

真正的问题从"怎么写好一个 skill？"变成了**"怎么架构一个 skill 系统，让 agent 能可靠地在其中导航？"**

这是拥有工具和拥有 workshop 之间的差距。是 agent *能做事* 和 agent *每次都按正确顺序做正确的事* 之间的差距。这篇文章讲的就是怎么建这个 workshop：把 skill 组织成 pipeline，给 agent 一个 CLI 来导航，在 ephemeral cloud session 里运行它们，让整个系统可靠到可以支撑团队的日常工作。

---

## 1. Agent Workflow 的三代演进

#### 直觉：从食谱卡片到厨房 brigade 系统

一个家庭厨师有食谱卡片——一次一道菜，需要时翻出来。一个餐厅厨房有 brigade 系统：工位、备料清单、时间安排、还有协调全局的 head chef。食谱没变，但食谱周围的*系统*才是让一晚出 200 个 cover 成为可能的东西。

Agent workflow 经历了类似的演进。

> **这篇文章和 loop engineering 有什么关系？** 2026 年 6 月，Peter Steinberger 那条"你应该设计让 agent 自动运行的 loop，而不是自己打 prompt"的推文（650 万浏览）引爆了 loop engineering 这个概念——核心理念是人不再手动输 prompt，而是构建一个外层循环，按计划触发 agent、检查结果、自动迭代。
>
> Loop engineering 和 skill pipeline 解决的是不同层面的问题。**Loop engineering 关注的是 agent *什么时候*、*多久一次*被触发**——替代手动 prompt 的自动化控制结构。**Skill pipeline 关注的是一次执行*内部*发生了什么**——多个 skill 怎么组织、连接、编排来完成任务。
>
> 两者是互补的：一个设计良好的 loop，它的循环体往往就是一个 skill pipeline。你需要两者——loop 负责外层自动化，pipeline 负责内层执行。本文聚焦在 pipeline 这一层。

### 第一代：Ad-Hoc Prompting（2023–2024）

每次交互都从零开始。你写 prompt，LLM 回复，你迭代。工具通过 function calling 手动调用（[Day 33](day33-tool-use.md)）。没有持久化，没有复用，没有系统。

- **优点**：最大灵活性
- **弱点**：零一致性——同样的任务，不同的 prompt，每次不同的结果
- **适用场景**：原型探索、一次性任务

### 第二代：Skill Library（2025）

[Day 40](day40-agent-skills.md) 标志着标准化 skill 的到来。你写一个 `SKILL.md`，agent 按需加载，skill 把 instructions + tools + knowledge 打包在一起。ClawHub 这样的 skill registry 让分享成为可能。

Agent 看到所有可用的 skill description，自己挑合适的。这是大多数团队现在的状态——有 skill library，agent 在其中自主导航。

- **优点**：可复用、可组合、可分享
- **弱点**：Agent 每次都要做 routing 决策。20 个 skill 时，它有时选错。多步 workflow 时，它半路丢线索
- **适用场景**：单能力构建、小规模 agent 部署

### 第三代：Orchestrated Skill Pipeline（2025–2026）

这是生产团队正在走的方向。与其让 agent 每次在 20 个 skill 里自由导航，你定义 **pipeline**：为已知 workflow 预编排的 skill 序列。Agent 的职责从"想清楚用哪些 skill"变成"执行这个 pipeline，在每个节点做聪明的决策"。

这就像给新员工一个 toolbox 说"去修东西"（第二代）和给他一本 runbook 说"这种类型的问题，按这个顺序用这三个工具"（第三代）的区别。他们仍然需要判断力——但结构消除了大部分 routing 错误。

| 代际 | Agent 的工作 | 失败模式 | 适用场景 |
|------|------------|--------------|----------|
| 第一代：Ad-Hoc | 从零开始做所有事 | 不一致 | 原型探索 |
| 第二代：Skill Library | 每次自己选 skill | 规模化时的 routing 错误 | 单任务 agent |
| 第三代：Pipeline | 执行已知流程，在分支点做决策 | Pipeline 刚性 vs 灵活性权衡 | 生产环境中的团队 workflow |

**关键洞察**：第三代不是替代第二代，而是叠加在第二代*之上*。你仍然需要精心设计的 skill（第二代）。但你加了一层 pipeline orchestration，处理那 80% 可预测的工作，把 agent 的完全自主权留给那 20% 真正需要自适应推理的部分。

---

## 2. Skill Sprawl 问题

在构建 pipeline 之前，先理解为什么需要它。一个团队从 5 个 skill 增长到 20 个时，会撞上几堵墙：

### 2.1 Discovery 问题

20 个 skill description 加载到 context window 里，agent 每次都要玩配对游戏："这 20 个 skill 里哪个匹配这个请求？"有时两个 skill 看起来同样相关。有时没有哪个完全合适，agent 就选最近的——错了。

这就是 **skill description collision** 问题。当 `deploy-app` 和 `deploy-service` 的 description 很相似时，agent 就在猜。有时猜对。有时它给你部署了用户面前的 app，而你说的是后端微服务。

### 2.2 Sequencing 问题

很多真实任务需要多个 skill 按顺序执行：investigate → diagnose → fix → verify。在第二代的 skill library 里，agent 每次都要自己想清楚这个顺序。它可能 investigate 完直接去 verify，然后才试图 fix——顺序错了。或者跳过 diagnose 直接 fix，基于不完整信息。

人类用 runbook 和 SOP 解决这个问题。Agent 需要等价物。

### 2.3 State 问题

Skill A 产生一个结果。Skill B 需要这个结果作为输入。在第二代系统里，这个 state handoff 通过 agent 的 context window 发生——Skill A 的输出留在对话里，agent 把相关部分传给 Skill B。简单两步流程没问题。但以下情况会崩：

- Pipeline 有 6+ 步，context window 被塞满
- 某步产生大量输出（logs、reports）不该占着 context
- Agent 需要从第 4 步重试，不想重做第 1–3 步

这三个问题——discovery、sequencing 和 state——就是 pipeline 解决的。

---

## 3. Skill Pipeline 的解剖

Pipeline 是一组预定义的 skill 编排，包含明确的 input/output contract、state handoff 机制和 failure handling。让我们拆解这个架构。

### 3.1 三层结构

![Pipeline 架构](../zh/images/day45/pipeline-architecture.png)
*Figure 1: Skill pipeline 系统的三层。Routing layer 将 intent 匹配到 pipeline。Skill chain 执行序列。Resource layer 管理 session、state 和外部资源。*

**第一层 — Routing**：请求进来时，routing layer 决定哪个 pipeline（如果有的话）匹配用户意图。可以是 agent 驱动（agent 读请求选 pipeline）或 rule-based（关键词匹配、显式 CLI command）。实践中大多用 hybrid：已知 workflow 用显式 command，模糊情况用 agent 推理。

**第二层 — Skill Chain**：Pipeline 的核心。每个节点是一次 skill 执行，有定义好的输入、输出和转换条件。Chain 可以是线性的（A → B → C）、分支的（A → B 或 C，取决于条件）或并行的（A → [B 和 C 同时] → D）。

**第三层 — Resource Management**：让 pipeline 跑起来的基础设施：ephemeral agent session、用于 skill 间数据传递的 state store、外部服务连接、生命周期管理（拉起、执行、清理）。

### 3.2 一个具体例子：Deploy-and-Verify Pipeline

来点实际的。DevOps 团队有一个常见 workflow：部署服务、跑健康检查、验证部署、通知团队。作为 pipeline 长这样：

```
┌─────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐
│  Deploy  │───▶│ Health     │───▶│ Verify   │───▶│ Notify  │
│  Service │    │ Check      │    │ Endpoints│    │ Team    │
└─────────┘    └───────────┘    └──────────┘    └─────────┘
                      │
                      ▼ (如果不健康)
               ┌───────────┐
               │ Rollback  │
               │ & Alert   │
               └───────────┘
```

每个方框是一个 skill。箭头定义执行顺序和分支条件。Pipeline 编码了团队知识："deploy 之后一定跑 health check"，"health check 失败就自动 rollback"，"做完通知团队"。

没有这个 pipeline，agent 每次都要自己想出这个顺序。它可能跳过 health check，可能 verify 之前就通知了，可能不知道 rollback 是个选项。Pipeline 消除了这些失败模式。

### 3.3 Pipeline 定义格式

实际上怎么写 pipeline 定义？本质上，pipeline 就是 skill、它们顺序和互联关系的声明：

```yaml
# pipeline: deploy-and-verify
name: deploy-and-verify
description: Deploy a service, verify health, and notify the team
trigger:
  command: "deploy"          # 触发此 pipeline 的 CLI command
  intent_keywords: ["deploy", "ship", "release"]

steps:
  - id: deploy
    skill: service-deploy
    inputs:
      service_name: "${user.service_name}"
      environment: "${user.environment | default: staging}"
    outputs:
      deployment_id: "${result.deployment_id}"
      version: "${result.version}"

  - id: health_check
    skill: health-check
    depends_on: deploy
    inputs:
      service_name: "${user.service_name}"
      environment: "${user.environment}"
    outputs:
      healthy: "${result.healthy}"
      issues: "${result.issues}"
    on_failure:
      action: goto
      target: rollback

  - id: verify
    skill: endpoint-verify
    depends_on: health_check
    condition: "${health_check.healthy == true}"
    inputs:
      endpoints: "${deploy.endpoints}"
    outputs:
      verified: "${result.all_passed}"

  - id: rollback
    skill: service-rollback
    inputs:
      deployment_id: "${deploy.deployment_id}"
    outputs: {}

  - id: notify
    skill: team-notify
    depends_on: [verify, rollback]
    inputs:
      message: "${pipeline.summary}"
      channel: "${user.channel | default: ops}"
```

这个定义捕捉了一个资深工程师直觉上知道的东西——但让它变得可复现、可审计、可被任何 agent session 执行。

---

## 4. Pipeline 语境下的 Skill：Contract 层

在 [Day 40](day40-agent-skills.md) 里，skill 被定义为 `SKILL.md` + scripts + resources。在 pipeline 里，skill 需要更多东西：**显式的 input/output contract**。

### 4.1 为什么 Contract 重要

自由格式的 skill 在人或 agent 直接阅读 instructions 并即兴发挥时可以工作。在 pipeline 里，skill 是机械式串联的。Skill B 不会去读 Skill A 的 instructions——它期望特定的输入格式。如果 Skill A 输出 `{"status": "ok"}` 而 Skill B 期望 `{"healthy": true}`，pipeline 就断了。

Contract 让这一切显式化：

```markdown
# Skill: health-check

## Input Contract
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| service_name | string | yes | Name of the service to check |
| environment | string | yes | One of: staging, production |
| timeout | int | no | Check timeout in seconds (default: 30) |

## Output Contract
| Field | Type | Description |
|-------|------|-------------|
| healthy | boolean | Overall health status |
| issues | array[object] | List of specific issues found |
| latency_ms | int | Response latency in milliseconds |
```

有了 contract，pipeline orchestrator 可以在执行前验证输入，在定义时（而非运行时）发现 mismatch，并在需要时在步骤间转换输出。

### 4.2 State Handoff 策略

数据怎么从一个 skill 流到下一个？三种方法，各有取舍：

**方法 1 — Context Window 传递**

Skill A 的输出留在 agent 的 context window 里。Skill B 从那里读。

- *优点*：简单，不需要外部基础设施
- *缺点*：Context window 污染；大输出挤占 instructions；重试很麻烦
- *适用*：短 pipeline（2–3 步）、小输出

**方法 2 — 外部 State Store（Skill 自己通过共享存储协调）**

每个 skill 把输出写到共享 store（文件、数据库、key-value store）。Pipeline 在步骤间传递引用（ID、path），而非数据本身。**关键在于：每个 skill 自己负责知道去哪里读、往哪里写**——orchestrator 只是按顺序触发 skill。

```
Orchestrator: "先跑 deploy，再跑 health_check"

Skill deploy:
  → 从 pipeline context 读输入
  → 执行
  → 写 /state/pipeline-123/deploy.json  （skill 自己决定路径）

Skill health_check:
  → 读 /state/pipeline-123/deploy.json   （skill 自己知道去哪找）
  → 执行
  → 写 /state/pipeline-123/health.json
```

- *优点*：Context window 干净，支持重试（step 2 可以重读 step 1 的输出而不重跑 step 1），处理大数据
- *缺点*：Skill 之间必须约定存储路径和格式——Skill B 需要知道 Skill A 的输出路径和格式。这是隐式耦合，不会出现在任何 contract 里
- *适用*：同一团队编写所有 skill、存储约定稳定的场景

**方法 3 — Orchestrator 管理的 Handoff（Pipeline 负责数据传递）**

与方法 2 的关键区别：**skill 之间完全不知道彼此的存在。** 每个 skill 只声明自己的 input/output contract。Orchestrator 读取 Skill A 的输出，根据 pipeline 定义做字段映射，然后作为 typed input 注入 Skill B。Skill 从不直接碰共享存储。

```python
# Orchestrator 负责传递——skill 是纯函数
deploy_result = await execute_skill("service-deploy", inputs)

# Orchestrator 根据 pipeline 定义做字段映射：
#   pipeline 说：health_check.service_name = deploy.service_name
#   pipeline 说：health_check.environment = deploy.environment
health_result = await execute_skill("health-check", {
    "service_name": deploy_result.service_name,
    "environment": deploy_result.environment
})
```

这个区别在 skill 来自不同团队、或需要适配字段名时很重要。在方法 2 里，如果 Skill A 输出 `app_version` 而 Skill B 期望 `version`，你得改其中一个 skill。在方法 3 里，orchestrator 处理映射——两个 skill 都不用改。

- *优点*：Skill 完全解耦——每个只关心自己的 contract；orchestrator 层做字段映射处理 mismatch；contract 违反在 pipeline 定义时就能发现
- *缺点*：需要一个理解 skill contract 并执行映射的编排层
- *适用*：Skill 来自不同团队、跨框架 pipeline、或任何需要严格 contract 执行的系统

**怎么选 2 和 3**：如果你自己写所有 skill 且它们共享一致的数据格式，方法 2 更简单。如果 skill 来自不同团队、字段命名不同、或需要独立替换，方法 3 的解耦值得那点编排开销。

### 4.3 Failure Contract

Skill 失败了怎么办？没有计划的话，agent 要么盲目重试，要么彻底放弃，要么——最糟的——带着垃圾输入继续下一步。Failure contract 提前定义这些：

| 策略 | 适用场景 | 例子 |
|------|---------|------|
| **Retry** | 瞬时失败（网络、限流） | API timeout → backoff 重试 |
| **Fallback** | 有替代 skill 时 | 主部署工具失败 → 用 legacy 部署 |
| **Skip** | 非关键步骤 | 通知失败 → 记 warning，继续 |
| **Abort + Alert** | 关键步骤，无法继续 | Deploy 失败 → 停止 pipeline，alert on-call |
| **Human Escalation** | 需要判断的模糊失败 | Health check 不确定 → 暂停，找人 |

Failure contract 放在 pipeline 定义里，不在 skill 里：

```yaml
on_failure:
  action: retry          # retry | fallback | skip | abort | escalate
  max_retries: 3
  backoff_seconds: 5
  fallback_skill: legacy-deploy
  escalate_to: "#ops-oncall"
```

这个分离很重要：skill 不需要知道自己失败时会发生什么。Pipeline orchestrator 根据这一步在整体 workflow 中的角色来做决策。

---

## 5. CLI 作为 Agent 接口

在生产环境里，agent 需要一个可靠的方式来发现和调用 skill。一个**为 agent 设计的 CLI**——不是为人设计的——是最有效的模式之一。

### 5.1 Skill 不决定自己的接口

一个 skill 定义了*做什么、怎么做*——但不规定*怎么被调用*。同一个 skill 可以通过多种接口暴露给 agent：

| 接口 | Agent 怎么调用 | 适用场景 |
|------|---------------|---------|
| **CLI command** | Shell 执行：`mycli deploy --service api` | 自包含 pipeline、ephemeral session |
| **MCP server**（[Day 38](day38-mcp-model-context-protocol.md)） | 标准化的 tool protocol | 跨平台工具共享 |
| **Function call** | Agent runtime 内的原生 tool calling | 单 agent session 内使用的简单 skill |
| **Context injection** | Agent 直接读 `SKILL.md` 然后按指令执行 | 没有 script 的纯推理类 skill |

CLI 只是其中一种选择——但它是目前最适合 skill pipeline 的那种。原因如下。

### 5.2 为什么是 CLI 而不是 API？

听起来反直觉。都 2026 年了，为什么不用 REST API？因为 agent 天生是 tool-caller（[Day 33](day33-tool-use.md)），2026 年的 agent 主要通过三种方式与世界交互：

1. **Function/tool calling** — 通过模型原生 tool interface 的结构化 API call
2. **Shell command** — 执行 CLI 工具并读 stdout
3. **MCP server**（[Day 38](day38-mcp-model-context-protocol.md)）— 标准化的 tool protocol

CLI 三者通吃：它可以被包装成 function call，可以直接在 shell 执行，也可以通过 MCP server 暴露。它是最通用的接口层。

但更重要的是，为 agent 设计的 CLI 有一些通用 API 没有的特性：

| 特性 | 对 Agent 为什么重要 |
|------|---------------------|
| **自文档化** | `--help` 输出就是内联文档；agent 不需要外部文档就能发现可用命令 |
| **可组合** | Unix pipe（`|`）、redirect（`>`）、chain（`&&`）让 agent 自然地组合命令 |
| **无状态** | 每次调用独立——完美适配 ephemeral session |
| **可脚本化** | Skill 可以作为 CLI subcommand 发布，复杂 skill 可以 bundle 多个 script |
| **结构化输出** | `--json` 给 agent 机器可读的结果 |

### 5.3 Agent-Facing CLI 的设计原则

为人设计的 CLI 优化可读性和交互性。为 agent 设计的 CLI 优化可预测性和机器可解析性。设计原则不同：

**原则 1：Command 对应 Skill（或 Pipeline）**

```
mycli deploy --service api --env staging     # → deploy-and-verify pipeline
mycli investigate --service api --symptom high-latency  # → investigate skill
mycli report --range 7d --format json        # → weekly-report pipeline
```

每个 command 是一个 skill 或 pipeline 的直接入口。Agent 不需要浏览 skill catalog——CLI 的 command 结构*就是* catalog。

**原则 2：默认结构化输出**

```bash
$ mycli deploy --service api --env staging --format json
{
  "status": "success",
  "deployment_id": "dep_abc123",
  "version": "v2.4.1",
  "endpoints": ["https://api-staging.example.com"],
  "duration_seconds": 47
}
```

Agent 读结构化 JSON，不是自由文本。这消除了 parsing error——agent 与 CLI 工具交互时最常见的失败源。

**原则 3：显式 Exit Code**

```
0  → 成功
1  → 通用失败
2  → 输入无效
3  → 依赖缺失
4  → 部分成功（pipeline 完成但有 warning）
10 → skill 未找到
```

Agent 检查 `$?` 来决定下一步。Exit code 2 意思是"修复你的参数"，不是"重试同样的东西"。

**原则 4：Skill Script 内联打包**

CLI 内部打包 skill script。当 pipeline 步骤调用 `health-check` skill 时，CLI 执行内联的 `health_check.sh` 或 `health_check.py`——不是外部 URL。这消除了一类网络故障，让 CLI 成为自包含的，可以部署到任何 cloud agent 环境。

```
mycli/
├── mycli                    # 入口
├── skills/
│   ├── service-deploy/
│   │   ├── SKILL.md         # Agent 指令
│   │   ├── deploy.sh        # 脚本
│   │   └── README.md        # 参考材料
│   ├── health-check/
│   │   ├── SKILL.md
│   │   ├── check.py
│   │   └── references/
│   │       └── check-spec.md
│   └── team-notify/
│       ├── SKILL.md
│       └── notify.py
└── pipelines/
    ├── deploy-and-verify.yaml
    └── investigate.yaml
```

这个结构呼应了 Day 40 的 skill 格式（`SKILL.md` + scripts + resources），但加了 pipeline 定义和统一的 CLI 入口。

---

## 6. Cloud Agent Session：Ephemeral 设计

我们描述的 pipeline 需要一个运行场所。对于 2026 年构建 workflow 系统的团队来说，主流模式是 **ephemeral cloud agent session**——按需拉起、执行 pipeline、然后释放的 session。

### 6.1 为什么不用持久化 Session？

一个持久化的 agent session——7×24 活着、不断累积 context——听起来不错。Agent "记住"了一切。但实践中，持久化 session 会带来问题：

| 问题 | 为什么重要 |
|------|-----------|
| **Context 膨胀** | 几天/几周后，context window 填满旧对话、过期 state 和无关历史——degrade agent 的决策质量 |
| **State 污染** | 一次 pipeline 执行中的瞬时错误会污染下一次的 session state |
| **安全攻击面** | 长寿命 session 累积 credential、token 和 access——一次 compromise 暴露一切 |
| **成本** | 空闲 agent session 仍然产生基础设施费用（模型预热、state 存储、监控） |
| **Debug 困难** | 出问题时，session 的漫长历史让人很难隔离触发失败的原因 |

Ephemeral session 解决了所有这些。每次 pipeline 执行（或一组相关执行）获得一个干净的 session。干净的 context。隔离的 state。有限的生命周期。Pipeline 完成（或失败）后，session 释放。

### 6.2 Session 生命周期

一个典型的 ephemeral session pipeline 执行生命周期：

```
1. 请求到达 → Router 选择 pipeline
2. Agent provider 配置新 session
3. Session 加载相关 skill（基于 pipeline 定义）
4. Pipeline 执行：逐 skill 执行，state 在步骤间传递
5. Session 捕获结果、日志、trace
6. Session 释放 → 资源回收
```

**Agent provider**——管理 session 生命周期的平台——负责步骤 2、5、6。团队负责步骤 1、3、4：定义 pipeline、skill 和 routing 逻辑。

### 6.3 跨 Session 的 State

如果 session 是 ephemeral 的，系统怎么在执行之间记住东西？三种模式：

**模式 1 — 无状态（纯函数）**

每次 pipeline 执行完全独立。不记得过去的运行。最简单可靠，适合真正自包含的 workflow：部署、健康检查、报告生成。

**模式 2 — 外部 Checkpoint**

Pipeline 把持久 state 写到外部 store（数据库、文件系统、对象存储）。下一次执行从这个 store 读。Session 本身是 ephemeral 的，但*工作产物*持久化。

```
Pipeline Run 1 → 写 deployment_log.json → session 释放
Pipeline Run 2 → 读 deployment_log.json → session 释放
```

**模式 3 — Session Affinity（软 State）**

对于需要短期连续性的 pipeline（比如跨越几小时的多步调查），agent provider 可以提供 session affinity：在有限时间内把相关请求路由到同一个 session，然后释放。这提供了 session 连续性的好处，又没有长期成本。

| 模式 | 复杂度 | 适用场景 |
|------|--------|---------|
| 无状态 | 低 | 独立任务（deploy、check、report） |
| 外部 Checkpoint | 中 | 需要历史的多步 workflow |
| Session Affinity | 高 | 交互式多轮 workflow |

大多数团队 workflow 适合模式 1 或 2。模式 3 留给真正需要对话连续性的调查或探索性任务。

---

## 7. Pipeline 编排模式

不是所有 pipeline 结构都一样。根据 workflow 的性质，不同的编排模式更合适。

### 7.1 Fixed DAG（有向无环图）

Pipeline 是固定的步骤序列，有定义好的分支条件。Agent 按顺序执行每一步，遵循分支。这是最可预测的模式——你明确知道哪些 skill 会运行、以什么顺序。

```
Deploy → Health Check → [健康? → Verify → Notify]
                      → [不健康? → Rollback → Alert]
```

- **最适合**：标准化的、可重复的 workflow（部署、CI/CD、报告生成）
- **优点**：可预测性、可审计性、容易测试
- **弱点**：不灵活——无法应对分支条件没覆盖到的情况

### 7.2 Agent-Routed（自主导航）

Agent 看到所有可用 skill，根据情况动态决定执行路径。没有固定序列——agent 每步之后评估状态，选择下一个动作。

```
Agent 收到: "API 返回 500 错误"
Agent 决定:
  1. 检查最近部署 (skill: deploy-history)
  2. 查看错误日志 (skill: log-analyzer)
  3. 找相关 commit (skill: git-bisect)
  4. 回滚 (skill: service-rollback)
  5. 验证修复 (skill: endpoint-verify)
```

- **最适合**：调查性任务、新颖问题、探索性工作
- **优点**：适应任何情况，处理 edge case 和新颖性
- **弱点**：非确定性——同样输入可能产生不同执行路径；难以审计；需要更强的模型推理

### 7.3 Hybrid（结构骨架 + Agent 自主）

这是大多数生产团队最终收敛到的模式。Pipeline 定义一个**骨架**——必须执行的步骤和决策点——但在每一步，agent 有自主权决定*怎么*执行这一步，并且可以在需要时插入额外步骤。

```
Pipeline 骨架:
  1. 调查 (agent 自己决定怎么查: 日志? 指标? 最近变更?)
  2. 诊断 (agent 自己决定怎么诊: 对比已知问题? 查 DB? 推理?)
  3. 修复 (agent 自己决定怎么修: rollback? patch? config change?)
  4. 验证 (强制: 必须确认修复有效)
  5. 记录 (强制: 写 incident summary)
```

骨架确保关键步骤不被跳过（验证、记录）。Agent 在每一步的自主权处理真实世界情况的多样性。

- **最适合**：大多数"大体可预测、偶尔有惊喜"的团队 workflow
- **优点**：平衡可靠性和适应性
- **弱点**：设计更复杂——需要识别哪些步骤该固定、哪些该放开

### 7.4 对比

| 维度 | Fixed DAG | Agent-Routed | Hybrid |
|------|-----------|-------------|--------|
| 可预测性 | 高 | 低 | 中 |
| 灵活性 | 低 | 高 | 中高 |
| 设计成本 | 低（定义一次） | 低（让 agent 自己搞） | 高（平衡结构和自主） |
| 可审计性 | 高 | 低 | 中 |
| 模型要求 | 任意模型 | 强推理模型 | 中等偏强模型 |
| 失败模式 | 刚性 | 不可预测 | 设计复杂度 |

**实践建议**：从 Fixed DAG 开始，处理最常见的 workflow。随着遇到 DAG 处理不了的 edge case，逐步向 Hybrid 演进。Agent-Routed 留给真正无法预判流程的新颖任务。

---

## 8. Skill Pipeline 的可靠性

在 [Day 41](day41-reliability-issues.md) 里，我们学到多步 agent pipeline 受困于 compound error：如果每步 95% 可靠，10 步 pipeline 只有 60% 的成功率。Pipeline 没有消除这个数学——但它给了你结构化工具来管理它。

### 8.1 幂等 Skill 设计

幂等 skill 无论执行一次还是十次都产生同样的结果。`deploy --version v2.4.1` 要么部署那个版本（如果还没部署），要么报告它已经在运行。跑五次不会部署五个副本。

幂等性是可靠 pipeline 的基础，因为它让重试变得安全：

```python
# 幂等的部署 skill
def deploy(service, version, environment):
    current = get_current_version(service, environment)
    if current == version:
        return {"status": "already_deployed", "version": version}
    # ... 执行部署
    return {"status": "deployed", "version": version}
```

没有幂等性，重试失败的步骤可能导致双重执行的副作用——双重扣费、重复记录、冲突部署。有了幂等性，重试是免费的。

### 8.2 每个 Skill 一个 Checkpoint

每次 skill 执行在下一步开始前把结果写到持久 checkpoint。如果第 4 步失败，从第 4 步的 checkpoint 恢复——而不是从头开始。

```python
class PipelineExecutor:
    def __init__(self, pipeline_def, state_store):
        self.pipeline_def = pipeline_def
        self.state_store = state_store
    
    async def execute(self, pipeline_id, inputs):
        # 从之前的运行加载 checkpoint
        completed = self.state_store.get_completed_steps(pipeline_id)
        
        for step in self.pipeline_def.steps:
            if step.id in completed:
                # 跳过已完成的步骤，加载结果
                result = self.state_store.get_result(pipeline_id, step.id)
            else:
                result = await self.run_step(step, inputs)
                self.state_store.save_checkpoint(
                    pipeline_id, step.id, result
                )
            inputs = {**inputs, **result}
        
        return self.state_store.get_all_results(pipeline_id)
```

这个模式直接借鉴自数据 pipeline 编排工具（Airflow、Prefect、Dagster）——在大规模场景下被验证多年。Agent pipeline 面临同样的可靠性挑战，受益于同样的解决方案。

### 8.3 Dry-Run 模式

在真正执行 pipeline 之前，agent（或人类 reviewer）可以触发 dry run：走过每一步，验证输入、检查依赖，但不执行有副作用的操作。

```bash
$ mycli deploy --service api --env production --dry-run
[DRY RUN] Step 1: deploy service 'api' version 'v2.4.1' to 'production'
[DRY RUN] Step 2: health check 'api' in 'production' (would check 5 endpoints)
[DRY RUN] Step 3: verify endpoints ['https://api.example.com/health', ...]
[DRY RUN] Step 4: notify '#ops' channel
[DRY RUN] Pipeline would complete in ~4 steps. No changes made.
```

Dry-run 是廉价的保险。它在 contract mismatch、缺失依赖和逻辑错误触及生产之前就捕获它们。

### 8.4 Pipeline 级可观测性

每次 skill 执行应该输出结构化 trace 数据：输入、输出、持续时间、token 使用量、成功/失败。这些数据有两个用途：

1. **调试**：Pipeline 失败时，你可以精确追踪哪一步失败以及为什么
2. **优化**：识别哪些步骤持续慢或贵

[Day 41](day41-reliability-issues.md) 介绍了 tracing 工具（Langfuse、Phoenix、Datadog LLM Observability）。在 pipeline 语境下，关键的附加项是 **trace correlation**：同一次 pipeline 执行的所有步骤共享一个 `pipeline_run_id`，所以你可以把整个执行看作一棵 trace 树。

```
Pipeline Run #abc123 (deploy-and-verify)
├── deploy        [✓ 47s]  service=api, version=v2.4.1
├── health_check  [✓ 12s]  healthy=true
├── verify        [✓ 8s]   endpoints=3/3 passed
└── notify        [✓ 1s]   channel=#ops
```

这个视图让你立刻看到发生了什么，不需要翻 individual skill 的日志。

---

## 9. 实践经验

构建过 skill pipeline 系统的团队报告了一致的经验教训：

1. **从无聊的事情开始**。最高价值的 pipeline 是最重复、最没意思的任务——部署、检查、报告。不是花哨的 AI 驱动分析。
2. **已知 workflow 用 fixed pipeline 胜过自主 agent**。Agent "自己想办法"的自由在 workflow 已经很清楚时是负债。结构赢。
3. **Contract 就是一切**。一旦你放松 input/output contract，pipeline 就开始不稳。尽早执行 contract，即使 3 个人的团队觉得过度设计。
4. **Dry-run 模式拯救职业生涯**。第一次有人不小心触发了生产部署 pipeline 时，你会感谢 `--dry-run`。
5. **从第一天起就有可观测性**。不要在第一次 incident 后才加 tracing。要在那之前。当（不是如果）pipeline 失败时，你需要 trace 来调试。

---

## 10. 更大的图景：未来的方向

从第二代（skill library）到第三代（orchestrated pipeline）的演进，呼应了软件工程中反复出现的模式：

- **Function → Library → Framework**：代码从单个 function 开始，组织成 library，然后出现 framework 来编码使用 library 的最佳实践。
- **Microservice → Service Mesh → Platform Engineering**：单个 service 被组织成 managed mesh，然后 platform team 在上面构建 self-service 系统。
- **Skill → Skill Library → Skill Pipeline**：单个 skill 被组织成 catalog，然后出现 pipeline 系统来编码团队 workflow。

我们现在正处于 skill pipeline 成为生产团队运行 agent-powered workflow 默认方式的阶段。工具仍然年轻——大多数团队自己搭建 pipeline orchestration——但模式正在收敛。

已经可见的下一个前沿是**跨团队 skill marketplace**：团队把 pipeline 和 skill 发布到内部 registry，其他团队发现并改编它们。CLI 不再只是一个团队的工具，而是组织能力的接口——agent 可调用、pipeline 编排、全公司标准化。

---

## 延伸阅读

### 相关课程
1. [Day 31: What Is an AI Agent?](day31-what-is-an-ai-agent.md) — Agent 架构基础
2. [Day 33: Tool Use](day33-tool-use.md) — Agent 如何调用外部 function
3. [Day 38: MCP](day38-mcp-model-context-protocol.md) — 标准化的 tool 连接协议
4. [Day 40: Agent Skills](day40-agent-skills.md) — 如何设计单个 skill
5. [Day 41: Reliability Issues](day41-reliability-issues.md) — Compound error 及应对工程方法
6. [Day 44: Human-AI Collaboration](day44-human-ai-collaboration.md) — 何时在 agent workflow 中引入人类

### 工具和框架
7. [OpenClaw Skills Documentation](https://docs.openclaw.ai/tools/skills) — Skill 格式、加载和 pipeline 配置
8. [AgentSkills.io Specification](https://agentskills.io) — 跨框架 skill 标准
9. [Prefect](https://prefect.io) / [Dagster](https://dagster.io) / [Airflow](https://airflow.apache.org) — 数据 pipeline 编排工具，其模式启发了 agent pipeline 设计

---

## 思考题

1. 想想你团队最常见的三个重复任务。它们能被表达为 Fixed DAG pipeline 吗？步骤是什么？哪里需要分支条件？

2. 什么时候应该让 agent 自己选 skill（agent-routed），什么时候应该预定义序列（fixed pipeline）？什么信号告诉你一个 workflow 太不可预测、不适合 pipeline 化？

3. 如果你的团队明天就要设计一个 agent-facing CLI，command 结构会是什么样的？第 5.2 节的哪些原则在你当前环境中最难实现？为什么？

---

## 总结

| 概念 | 一句话解释 |
|------|-----------|
| Skill Pipeline | 为已知 workflow 预编排的 skill 序列 |
| 三代演进 | Ad-hoc prompting → Skill library → Orchestrated pipeline |
| Skill Sprawl | 20 个 skill 没有组织时发生的事 |
| Input/Output Contract | Skill 接受什么、返回什么的显式 schema |
| State Handoff | 数据在 pipeline 步骤间流动的方式（context、外部 store、结构化 object） |
| Failure Contract | Skill 失败时的预定义行为（retry、fallback、skip、abort、escalate） |
| Agent-Facing CLI | 为 agent 调用设计的 CLI：结构化输出、显式 exit code、内联 script |
| Ephemeral Session | 按 pipeline 拉起、用完即释放的 cloud agent session |
| Fixed DAG vs Agent-Routed vs Hybrid | 三种编排模式，在可预测性和灵活性间取舍 |
| 幂等 Skill | 重复执行产生同样结果的 skill——安全重试的基础 |

**核心要点**：单个 skill 给 agent 能力。Pipeline 给 agent 可靠性。从第二代到第三代的旅程——从 skill library 到 orchestrated system——是从"agent 能做事"到"agent 每次都按正确顺序做正确的事"的旅程。从你最无聊、最重复的 workflow 开始，把它们包进 fixed pipeline，给 agent 一个 CLI 来导航，然后从那里迭代。

---

*Day 45 of 60 | LLM Fundamentals*
*Word count: ~3400 | Reading time: ~16 minutes*
