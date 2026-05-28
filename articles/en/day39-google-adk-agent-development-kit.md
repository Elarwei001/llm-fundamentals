# Day 39: Google ADK — The Agent Development Kit

> **Core Question**: What is Google's Agent Development Kit, and why is it becoming a first-class framework for building production AI agents?

---

## Opening

Imagine you're building a restaurant. You could buy raw lumber, forge your own nails, and hand-carve every chair — or you could rent a fully equipped commercial kitchen with standardized workstations, prep stations already plumbed, and health-code-compliant ventilation. That's the difference between wiring agents together from scratch and using a framework like Google ADK.

Before 2025, if you wanted to build an AI agent — something that could reason, call tools, maintain memory, and maybe coordinate with other agents — you were essentially a carpenter. You glued together LangChain chains, wrote custom orchestration loops, managed session state by hand, and prayed your error handling was good enough. It worked, but every project reinvented the same plumbing.

In April 2025, Google released the Agent Development Kit (ADK) as open source, offering a different proposition: what if the framework handled the infrastructure — sessions, events, tools, deployment, observability — and you just focused on what your agent *does*?

By May 2026, ADK has matured through a rapid release cycle to version 2.0, added support for Python, Java, Go, Kotlin, and Android, and introduced a graph-based Workflow Runtime that competes directly with LangGraph. Let's look at what it is, how it works, and where it fits in the agent framework landscape.

---

## 1. What is Google ADK?

### Intuition: The Operating System for Agents

Think of ADK as an operating system specifically designed for AI agents. Just as your laptop's OS manages processes, memory, file system access, and inter-process communication without you writing low-level code, ADK manages agent lifecycles, tool calls, session state, inter-agent communication, and deployment — without you wiring it all by hand.

You write the "application" (what the agent should do), and ADK provides the "kernel" (everything that makes it run reliably in production).

### Key Facts

Google ADK is an **open-source, code-first, event-driven framework** for building, evaluating, and deploying AI agents. It was first released in April 2025 and reached a major milestone with **ADK Python 2.0 GA on May 19, 2026** ([GitHub: google/adk-python](https://github.com/google/adk-python)).

Core principles:
- **Code-first**: You write Python (or Java, Go, Kotlin), not YAML or visual drag-and-drop
- **Event-driven**: Everything — tool calls, state changes, agent transitions — flows through events
- **Model-flexible**: Optimized for Gemini but supports 200+ models via Vertex AI Model Garden and others via LiteLLM
- **Production-ready**: Built-in OpenTelemetry, evaluation frameworks, one-command Cloud Run deployment

![Figure 1: Google ADK layered architecture showing the Agent Layer, Runtime & Orchestration, and Tool & Integration Layer](../zh/images/day39/adk-architecture-overview.png)
*Figure 1: ADK's layered architecture. Agents sit on top, the runtime orchestrates execution in the middle, and tools/integrations live at the bottom.*

### Architecture Layers

ADK is organized into three clean layers:

| Layer | Components | Responsibility |
|-------|-----------|---------------|
| **Agent Layer** | LlmAgent, WorkflowAgent, Custom Agent | Define *what* the agent does and how it reasons |
| **Runtime & Orchestration** | Runner, Session Service, Event Bus | Manage *how* agents execute, persist state, and stream events |
| **Tool & Integration Layer** | MCP Tools, Custom Tools, Built-in Tools | Connect agents to the outside world |

Two cross-cutting concerns span all layers:
- **Deployment**: Cloud Run, Google Kubernetes Engine (GKE), or local execution
- **Observability**: OpenTelemetry tracing, ADK Dev UI for real-time debugging

---

## 2. The Three Agent Types

### Intuition: Three Kinds of Workers

Imagine you're running a project team. Sometimes you need a **specialist** — one person who handles conversations and makes decisions (LlmAgent). Sometimes you need an **assembly line** — a sequence of workers where each does one step and passes the result along (SequentialAgent). And sometimes you need a **task force** — multiple people working on different parts of the same problem simultaneously (ParallelAgent).

Google ADK gives you exactly these patterns, plus a LoopAgent for iterative refinement.

### 2.1 LlmAgent — The Conversational Core

The `LlmAgent` is the workhorse. It wraps an LLM with instructions, tools, and optional sub-agents:

```python
from google.adk import Agent

research_agent = Agent(
    name="research_agent",
    model="gemini-2.5-flash",
    instruction="""You are a research assistant. 
    Use search tools to find information and provide 
    cited, accurate answers.""",
    tools=[google_search, arxiv_lookup],
)
```

When you call `agent.run()`, ADK:
1. Sends the user message + instructions + tool definitions to the LLM
2. If the LLM requests a tool call, ADK executes it and feeds the result back
3. Repeats until the LLM produces a final response
4. Streams all intermediate events (tool calls, reasoning, partial outputs) through the event bus

#### Intuition: The Manager with a Rolodex

An LlmAgent is like a senior manager who has a big Rolodex of specialists (tools). When someone asks a question, the manager thinks about whether they can answer directly or whether they need to call a specialist. If they call a specialist, they wait for the result, then synthesize it into a response.

### 2.2 Workflow Agents — Structured Orchestration

ADK 2.0 introduced the **Workflow API**, a graph-based execution engine that lets you compose deterministic flows. The three built-in workflow agents are:

![Figure 2: The three workflow agent types in ADK — SequentialAgent, ParallelAgent, and LoopAgent](../zh/images/day39/adk-workflow-agent-types.png)
*Figure 2: ADK's three workflow primitives. Sequential for step-by-step, Parallel for concurrent tasks, Loop for iterative refinement.*

**SequentialAgent** — runs sub-agents one after another, passing results forward:

```python
from google.adk import Agent, Workflow

# Define individual agents
extract_agent = Agent(name="extract", instruction="Extract key entities from the text.")
analyze_agent = Agent(name="analyze", instruction="Analyze the extracted entities for relationships.")
summarize_agent = Agent(name="summarize", instruction="Write a concise summary based on the analysis.")

# Compose as a sequential workflow
pipeline = Workflow(
    name="document_pipeline",
    edges=[("START", extract_agent, analyze_agent, summarize_agent)],
)
```

**ParallelAgent** — runs sub-agents simultaneously, useful when tasks are independent:

```python
parallel_check = Workflow(
    name="parallel_verification",
    edges=[
        ("START", [fact_check_agent, grammar_check_agent, tone_check_agent]),
        ([fact_check_agent, grammar_check_agent, tone_check_agent], "END"),
    ],
)
```

**LoopAgent** — runs sub-agents iteratively until a condition is met, perfect for refinement and self-correction:

```python
refinement_loop = Workflow(
    name="code_review_loop",
    edges=[("START", write_code, review_code)],
    # Loop back if review finds issues
    loop_until=lambda state: state.get("review_passed") == True,
)
```

### 2.3 The Task API — Agent-to-Agent Delegation

ADK 2.0 also introduced the **Task API**, which enables structured delegation between agents. This is different from workflows — it's about one agent *delegating work* to another, not about predetermined flow.

```python
# A coordinator delegates to specialists
coordinator = Agent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="Delegate tasks to the appropriate specialist.",
    sub_agents=[code_agent, math_agent, writing_agent],
)
```

The Task API supports three delegation modes:

| Mode | Behavior | Analogy |
|------|----------|---------|
| **Chat** | Full conversation with sub-agent, manual return | "Transfer the caller to the specialist" |
| **Task** | Goal-oriented execution, auto-return when done | "Ask the specialist to handle this and report back" |
| **Singleton** | Persistent agent maintaining context across turns | "Your dedicated account manager" |

---

## 3. The Bigger Picture: A2A, MCP, and the Agent Ecosystem

Google ADK doesn't exist in isolation. It's part of a three-protocol stack that Google is building for the agentic web:

![Figure 3: The Google agent ecosystem showing ADK, A2A Protocol, and MCP together](../zh/images/day39/adk-a2a-mcp-ecosystem.png)
*Figure 3: ADK sits in the center, connecting to LLMs for reasoning, MCP for tools, A2A for other agents, and Google Cloud for deployment.*

### 3.1 MCP — Agent-to-Tool Communication

The **Model Context Protocol (MCP)**, which we covered in [Day 38](day38-mcp-model-context-protocol.md), is how agents connect to external tools and data sources. ADK has **native MCP support** — you can plug in any MCP-compatible tool server and your agents can use those tools immediately.

This is a big deal. Before MCP, every framework had its own tool integration format. With MCP, a tool built once works with any MCP-compatible agent framework.

### 3.2 A2A — Agent-to-Agent Communication

The **Agent2Agent Protocol (A2A)**, announced by Google in April 2025 ([Google Developers Blog](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)), is an open protocol for inter-agent communication. It defines:

- **Agent Cards**: JSON documents advertising what an agent can do
- **Tasks**: The unit of work exchanged between agents
- **Transport**: HTTP + Server-Sent Events (SSE) + JSON-RPC 2.0

#### Intuition: The Universal Translator

A2A is like a universal translator between agents. Imagine a French chef and a Japanese engineer need to collaborate on a project. They don't speak each other's language, but if they both speak English as a lingua franca, they can work together. A2A is that lingua franca for AI agents — regardless of whether they're built with ADK, LangGraph, or any other framework.

ADK 2.0 has **native A2A support**, meaning an ADK agent can discover and communicate with any A2A-compliant agent, even if it's running on a different server built with a different framework.

### 3.3 The Three-Protocol Stack

| Protocol | Scope | Analogy |
|----------|-------|---------|
| **MCP** | Agent → Tools | USB port (connect peripherals) |
| **A2A** | Agent → Agent | HTTP/API (connect services) |
| **ADK** | Agent lifecycle | Operating system (manage everything) |

Together, these three create a complete ecosystem: ADK builds and runs agents, MCP connects them to tools, and A2A lets them collaborate with each other.

---

## 4. Multi-Language Support and On-Device Agents

One of ADK's distinguishing features is its multi-language SDK strategy. As of May 2026:

| SDK | Status | Best For |
|-----|--------|----------|
| **Python** | v2.1.0 (GA) | General agent development, most features |
| **Java** | v1.0.0 (GA) | Enterprise backend integration |
| **Go** | v1.0.0 (GA) | High-performance, concurrent services |
| **Kotlin** | v0.1.0 (Preview) | Android app development, hybrid cloud+edge |
| **Android** | v0.1.0 (Preview) | On-device agents with Gemini Nano |

#### Intuition: Write Once, Deploy Everywhere (Almost)

Think of ADK's multi-language approach like how web frameworks exist in many languages. Express.js (Node), Flask (Python), and Gin (Go) all solve the same problem, but in the language your team already uses. ADK does the same for agents.

### On-Device Agents: The Edge Play

The **ADK for Android** announcement at Google I/O 2026 ([Google Developers Blog, May 21, 2026](https://developers.googleblog.com/adk-kotlin-android-building-ai-agents/)) is particularly interesting. It enables **hybrid orchestration**: a cloud-based orchestrator agent that can delegate specific sub-tasks to on-device agents running Gemini Nano.

```kotlin
// On-device sub-agents: run locally with Gemini Nano — private data never leaves the device
val onDeviceRetrievalAgent = LlmAgent(
    name = "on_device_retrieval",
    model = GeminiNano(),  // Gemini Nano — on-device model, no network required
    instruction = "Retrieve the user's booking confirmation emails and personal documents locally. Return structured itinerary information.",
    tools = listOf(EmailRetrievalTool(), DocumentParserTool()),
)

val validationAgent = LlmAgent(
    name = "on_device_validation",
    model = GeminiNano(),  // Also uses on-device model
    instruction = "Validate that the extracted itinerary information is complete and consistent. Check dates, flight numbers, hotel addresses.",
)

// Cloud orchestrator: delegates privacy-sensitive tasks to on-device sub-agents
val orchestrator = LlmAgent(
    name = "travel_assistant",
    model = Gemini(apiKey = apiKey, name = "gemini-2.5-flash"),
    instruction = "You are a travel assistant helping users manage their itineraries. Delegate privacy-sensitive data retrieval and validation to on-device agents.",
    tools = listOf(GetTripDetailsTool(tripId)),
    subAgents = listOf(onDeviceRetrievalAgent, validationAgent),
)
```

Why does this matter? Privacy-sensitive operations — like reading a user's booking confirmation email or parsing personal documents — can happen entirely on-device. The cloud agent never sees the raw data. It only receives the structured results from the on-device sub-agent.

This is a pattern we'll see more of as on-device LLMs like Gemini Nano (available on 140M+ Android devices) mature.

---

## 5. Developer Experience: The Dev UI and Evaluation

### Intuition: The Flight Simulator for Agents

Building agents is like flying an airplane — you need a dashboard that shows you what's happening in real time. ADK's Dev UI is that dashboard.

ADK comes with a built-in web-based developer tool that provides:

1. **Real-time execution visualization**: See every tool call, model reasoning step, and state change as it happens
2. **Visual graph view**: Map out your agent architecture, see which agent is active and why
3. **Structured trace view**: Dive into execution logs with filtering and search
4. **Session management**: View and resume past conversations, inspect session state
5. **Human-in-the-loop (HITL)**: Pause execution for human approval on critical decisions

Launch it with:
```bash
adk web path/to/agents_dir
```

### Built-in Evaluation Framework

ADK includes an evaluation framework for measuring agent performance:

```python
from google.adk.evaluation import AgentEvaluator

evaluator = AgentEvaluator(agent=my_agent)
results = evaluator.evaluate(
    test_cases=[
        {"input": "What's the weather in Tokyo?", "expected_tool": "weather_api"},
        {"input": "Book a flight to Paris", "expected_tool": "flight_booking"},
    ],
    metrics=["tool_accuracy", "response_quality", "latency"],
)
```

This matters because agent evaluation is genuinely hard (we'll cover this in Day 42). Having it built into the framework means you can set up CI/CD for your agents — automatically testing that tool calls are correct, responses are on-topic, and latency is acceptable — before every deployment.

---

## 6. Framework Comparison: Where ADK Fits

![Figure 4: Agent framework comparison across five dimensions (2026)](../zh/images/day39/agent-framework-comparison.png)
*Figure 4: How major agent frameworks compare in 2026. ADK excels at cloud integration and multi-agent support; LangGraph leads in community and model flexibility.*

| Framework | Orchestration Model | Model Support | Cloud Integration | Best For |
|-----------|-------------------|---------------|-------------------|----------|
| **Google ADK** | Hierarchical agent tree + graph workflows | Gemini-first, 200+ via Vertex AI | Native Google Cloud | Google Cloud teams, production multi-agent |
| **LangGraph** | Directed graph with conditional edges | Any model (Claude, GPT, open-source) | Pluggable (any cloud) | Complex workflows, maximum flexibility |
| **CrewAI** | Role-based crews with process types | Any model | Limited built-in | Quick prototyping, role-based collaboration |
| **Anthropic Agent SDK** | Explicit handoffs | Claude only | Anthropic API | Claude-centric applications |
| **OpenAI Agents SDK** | Handoff-based | OpenAI models | OpenAI API | OpenAI-centric applications |

Note: These frameworks target overlapping but distinct use cases. ADK's strength is production multi-agent systems on Google Cloud; LangGraph's strength is maximum flexibility and ecosystem; CrewAI's strength is simplicity and rapid prototyping. They are not directly "better or worse" — they're optimized for different contexts.

---

## 7. ADK in Practice: A Complete Example

Let's build a multi-agent research system that demonstrates ADK's key features:

```python
from google.adk import Agent, Workflow
from google.adk.tools import google_search, mcp_tool

# Step 1: Define specialist agents
search_agent = Agent(
    name="searcher",
    model="gemini-2.5-flash",
    instruction="Search for information and return raw findings with sources.",
    tools=[google_search],
)

analysis_agent = Agent(
    name="analyst",
    model="gemini-2.5-flash",
    instruction="Analyze the findings, identify patterns, and assess credibility.",
)

writer_agent = Agent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Write a clear, well-structured research brief based on the analysis.",
)

# Step 2: Compose into a sequential workflow
research_pipeline = Workflow(
    name="research_system",
    edges=[
        ("START", search_agent),        # Search first
        (search_agent, analysis_agent),  # Then analyze
        (analysis_agent, writer_agent),  # Then write
        (writer_agent, "END"),
    ],
)

# Step 3: Run the pipeline
result = research_pipeline.run("What are the latest advances in state space models?")
print(result.output)
```

This example shows:
- **Three specialist agents**, each with a clear single responsibility
- **Sequential composition** via the Workflow API
- **Automatic state passing** between stages (the output of one feeds into the next)
- **One command to deploy**: `adk deploy cloud-run research_system`

---

## 8. The Math Behind Agent Orchestration

For readers who want to understand the formal model:

An ADK agent can be viewed as a state machine. At each turn, the agent receives an observation and produces an action:

$$
\begin{aligned}
a_t &= \pi_\theta(s_t, \text{instruction}, \text{tools}) \\
s_{t+1} &= T(s_t, a_t, o_t)
\end{aligned}
$$

Where:
- **a_t** is the action at step t (tool call or final response)
- **s_t** is the session state (conversation history + persistent state)
- **instruction** is the system prompt
- **tools** are the available tool definitions
- **T** is the transition function (handled by the ADK runtime)
- **o_t** is the observation (tool result or user input)

For a **WorkflowAgent**, the orchestration is a directed graph:

$$
G = (V, E), \quad V = \{v_1, v_2, ..., v_n\}, \quad E \subseteq V \times V
$$

Each vertex **v_i** is an agent or Python function. Each edge **(v_i, v_j)** represents a dependency — **v_j** executes after **v_i** completes. Conditional edges add a predicate:

$$
(v_i, v_j, c) \in E_{\text{cond}} \implies \text{execute } v_j \text{ if } c(s) = \text{true}
$$

The runtime evaluates conditions against the shared workflow state, enabling branching, loops, and dynamic routing.

---

## 9. Common Misconceptions

### ❌ "ADK only works with Gemini"

While ADK is optimized for Gemini (and provides the best experience with Gemini 2.0/2.5), it supports 200+ models through Vertex AI Model Garden and integrates with Claude, GPT, Mistral, and others via LiteLLM. You're not locked in — you just get a smoother experience with Google's models.

### ❌ "ADK is just another LangChain wrapper"

ADK is a ground-up framework with its own runtime, event system, session management, and deployment pipeline. It's not a wrapper around LangChain. The philosophical difference is significant: LangChain/LangGraph prioritize flexibility and ecosystem breadth; ADK prioritizes opinionated production-readiness and Google Cloud integration.

### ❌ "ADK is only for Google Cloud"

You can run ADK agents locally with `adk run`, deploy to any Docker-compatible environment, or use the Agent Engine on Vertex AI. Google Cloud is the best-supported deployment target, but not the only one.

### ❌ "Workflow orchestration is no different from writing the process into Skills/Prompts"

They look similar on the surface — both are "giving the agent rules to follow." But there are three critical differences:

**1. Control lives in the runtime, not the compile time.** Workflows in Skills/Prompts are "compile-time" — the agent is essentially "reading a manual," and execution depth depends on the LLM's instruction-following ability. If something goes wrong mid-step (API down, unexpected format), the agent can only rely on its own reasoning to cope. ADK's Workflow Agents (SequentialAgent, ParallelAgent, LoopAgent) are **runtime engines** — they don't depend on the LLM "remembering" what step comes next. The framework enforces execution order. Error handling, timeouts, retries, and state persistence are all built in.

> Analogy: Skills are like "giving an employee a manual and hoping they follow it." Workflows are like "a conveyor belt on an assembly line — each station does its step automatically." One relies on discipline, the other on mechanism.

**2. State management and persistence.** Skills' "memory" is the context window — when it fills up, it's gone. Cross-session state requires custom solutions. ADK has Session Service for state persistence, Event Bus for tracking every execution step, LoopAgent iteration state maintained by the framework, and workflows that can be paused, resumed, and rolled back.

**3. Composition and extensibility.** Collaboration between skills relies on the agent judging "which skill should I use now." Parallel execution requires the agent to understand "I can do both things at once." ADK's ParallelAgent natively supports concurrency, LoopAgent supports conditional iteration, and the A2A protocol enables agents built with different frameworks to delegate tasks to each other.

| Scenario | Skills/Prompt | ADK Workflow |
|------|-------------|-------------|
| Simple single-step task | ✅ Sufficient | Overkill |
| Fixed 2-3 step flow | ✅ Works | Also fine, maybe heavier |
| Multi-step reasoning + conditional branching | ⚠️ Depends on LLM reasoning, unstable | ✅ Framework guarantees |
| Parallel execution needed | ❌ | ✅ ParallelAgent |
| Iteration + self-check needed | ⚠️ Depends on prompt engineering | ✅ LoopAgent |
| Persistence + resumability needed | ❌ | ✅ Session Service |
| Cross-framework/cross-service agent collaboration | ❌ | ✅ A2A native support |

**One-line summary**: Skills are the agent's "knowledge," Workflows are the agent's "skeleton." Skills tell the agent *how to think*, Workflows tell the system *how to execute*. The former depends on LLM reasoning, the latter on framework mechanisms. For simple tasks they look the same, but for complex, multi-step production scenarios requiring reliability, the gap goes from "hints" to "guarantees."

---

## 10. Frontier: What's New and What's Next

### Recent Developments (Last 6 Months)

| Date | Event | Significance |
|------|-------|-------------|
| **Dec 2025** | ADK for Java & Go v1.0.0 released | Multi-language expansion beyond Python |
| **Mar 2026** | ADK Python v1.20+ with mature APIs | API stability, improved session management |
| **Apr 2026** | A2A Protocol integration in ADK | Native agent-to-agent communication across frameworks |
| **May 15, 2026** | [ADK Python 2.0 GA](https://github.com/google/adk-python) released | Workflow Runtime, Task API, breaking changes from 1.x |
| **May 18, 2026** | [ADK Python v2.1.0 on PyPI](https://pypi.org/project/google-adk/) | Bug fixes, refined Workflow and Task APIs |
| **May 21, 2026** | [ADK for Kotlin & Android 0.1.0](https://developers.googleblog.com/adk-kotlin-android-building-ai-agents/) announced at Google I/O 2026 | On-device agents with Gemini Nano, hybrid cloud-edge orchestration |

### What to Watch

1. **ADK Python 2.x maturation**: The Workflow Runtime and Task API are brand new (May 2026). Expect rapid iteration as production users stress-test them.
2. **A2A Protocol adoption**: As more frameworks add A2A support, the promise of cross-framework agent interoperability becomes real. Watch the [A2A GitHub](https://github.com/a2aproject/A2A) for reference implementations.
3. **On-device agents**: ADK for Android is at v0.1.0, but the hybrid cloud-edge pattern — where a cloud orchestrator delegates to on-device sub-agents for privacy — could reshape how we think about agent architecture.
4. **Enterprise features**: Google is positioning ADK as the framework for enterprise agent deployments on Google Cloud. Expect more built-in compliance, audit logging, and governance features.

---

## 11. Further Reading

### Beginner
1. [ADK Official Documentation](https://google.github.io/adk-docs/) — The best starting point with tutorials and API reference
2. [Google ADK GitHub Repository](https://github.com/google/adk-python) — Source code, samples, and contributing guide
3. [Building Smart in 2026: A Hands-On First Look at Google's ADK](https://dev.to/njericodecraft/building-smart-in-2026-a-hands-on-first-look-at-googles-agent-development-kit-adk-3n0) — Practical walkthrough for first-timers

### Advanced
1. [The Complete Guide to Google's ADK](https://sidbharath.com/blog/the-complete-guide-to-googles-agent-development-kit-adk/) — Deep architectural tour
2. [ADK 2.0: From Chatbots to Collaborative Deterministic AI Workflows](https://dr-arsanjani.medium.com/adk-2-0-from-chatbots-to-collaborative-deterministic-ai-workflows-c8656f3beab4) — In-depth look at the Workflow Runtime
3. [Multi-Agent Deployment with ADK and GKE](https://medium.com/google-cloud/multi-agent-deployment-with-the-agent-development-kit-adk-gke-gke-mcp-server-and-gemini-cli-f517ea7436db) — Production deployment patterns

### Papers & Specifications
1. ["Agent2Agent Protocol (A2A)" Specification](https://a2a-protocol.org/latest/) — The open protocol for inter-agent communication
2. ["Model Context Protocol" Specification](https://modelcontextprotocol.io/) — The standard for agent-to-tool communication
3. ["The Agent Framework Wars: Google ADK vs LangGraph vs CrewAI vs Anthropic Agent SDK"](https://1337skills.com/blog/2026-04-17-agent-framework-wars-google-adk-langchain-crewai-comparison/) — Comprehensive 2026 comparison

---

## Reflection Questions

1. Why do you think Google chose an *event-driven* architecture for ADK rather than a simpler request-response model? What capabilities does it enable?
2. If you were building a multi-agent customer support system, would you choose ADK's hierarchical agent tree or LangGraph's graph-based model? What trade-offs would you consider?
3. The hybrid cloud-edge pattern (cloud orchestrator + on-device sub-agents) creates interesting privacy implications. What new types of applications does this enable that weren't possible before?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| **ADK** | Google's open-source framework for building, evaluating, and deploying AI agents |
| **LlmAgent** | The core agent type that wraps an LLM with tools and instructions |
| **WorkflowAgent** | A graph-based agent that orchestrates sub-agents in sequential, parallel, or loop patterns |
| **Task API** | Structured agent-to-agent delegation with chat, task, and singleton modes |
| **Runner** | The execution engine that manages sessions, events, and tool orchestration |
| **A2A Protocol** | Open standard for agent-to-agent communication across frameworks |
| **MCP Integration** | Native support for Model Context Protocol tools |
| **ADK Dev UI** | Built-in web tool for real-time debugging, tracing, and evaluation |
| **On-device agents** | ADK for Android enables hybrid cloud-edge orchestration with Gemini Nano |

**Key Takeaway**: Google ADK represents Google's bet that AI agent development needs the same kind of opinionated, production-ready framework that web development got with Django or Rails. It's not the most flexible option — LangGraph wins there — but it offers the smoothest path from prototype to production if you're in the Google ecosystem. The real story is the three-protocol stack (ADK + MCP + A2A) that could define how agents are built, equipped, and interconnected across the industry.

---

*Day 39 of 60 | LLM Fundamentals*
*Word count: ~2800 | Reading time: ~14 minutes*
