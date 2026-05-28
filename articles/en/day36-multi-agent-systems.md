# Day 36: Multi-Agent Systems

> **Core Question**: When multiple AI agents work together, why does adding more agents often make things worse — and how do you design systems where collaboration actually helps?

---

## Opening

Imagine you're managing a team of five people. If everyone knows their role, communication is clear, and there's a shared understanding of the goal, the team can accomplish far more than any individual. But if roles are vague, people talk over each other, and nobody checks the work — five people can produce a worse result than one person working alone.

Multi-agent systems in AI follow exactly the same pattern. The idea is seductively simple: instead of one LLM doing everything, let multiple specialized agents collaborate. One researches, another analyzes, a third writes, and a fourth reviews. The promise is emergent capability — the system is smarter than any single agent.

The reality is more complicated. Research from UC Berkeley's MAST study (Cemri et al., 2025) cataloged **14 distinct failure modes** across 1,600+ execution traces of multi-agent LLM systems. A 2026 study on "Agent Drift" found that coordination between agents degrades over time in predictable ways. The "bag of agents" approach — just throwing more agents at a problem — can produce error rates **17x higher** than a single well-designed agent.

Today we'll understand why multi-agent systems are both the most exciting and most dangerous pattern in AI engineering. We'll cover the coordination topologies, the failure modes, the emerging protocol standards, and when multi-agent actually helps versus when it hurts.

---

## 1. Why Multi-Agent?

#### Intuition: The Restaurant Kitchen

Think of a single-agent LLM like a solo chef cooking every dish in a restaurant — appetizers, mains, desserts. A multi-agent system is like a full kitchen brigade: a sous-chef coordinates, a grill cook handles proteins, a pastry chef handles desserts, and a expeditor checks plates before they leave. The brigade is faster and produces better food — but only if the communication between stations works. If the grill cook doesn't hear the order change, the wrong dish goes out.

### 1.1 The Motivation

Single LLM agents hit three fundamental limits:

| Limit | Description | Multi-Agent Response |
|-------|-------------|---------------------|
| **Context window** | One agent can only hold so much information | Different agents maintain different context |
| **Skill specialization** | No single prompt makes an LLM great at everything | Each agent gets an optimized system prompt |
| **Error correction** | An agent can't easily spot its own mistakes | A reviewer agent provides independent verification |

The key insight is **separation of concerns** — a principle borrowed from software engineering. Instead of one monolithic agent trying to do everything, you decompose the task into specialized roles.

### 1.2 When It Works

Multi-agent systems shine in tasks that are:

- **Decomposable**: The task can be broken into independent subtasks (research + writing + review)
- **Verifiable**: Outputs can be checked by another agent (code review, fact-checking)
- **Parallelizable**: Subtasks can run concurrently, reducing total latency
- **Specialized**: Different subtasks benefit from different prompting strategies or even different models

### 1.3 When It Doesn't

Multi-agent struggles when tasks are:

- **Tightly coupled**: Each step depends heavily on the previous step's exact output
- **Simple**: The overhead of coordination exceeds the benefit of specialization
- **Ambiguous**: If the task itself is unclear, more agents just amplify the confusion

---

## 2. Coordination Topologies

![Figure 1: Three common multi-agent coordination topologies — Star/Hub, Pipeline/Sequential, and Mesh/Peer-to-Peer](../zh/images/day36/multi-agent-topologies.png)
*Figure 1: Three fundamental coordination topologies. Each has different trade-offs in control, latency, and resilience.*

The way agents communicate determines everything about the system's behavior. There are three fundamental patterns.

### 2.1 Star (Hub-and-Spoke)

In a star topology, a central **orchestrator agent** receives the user request, decomposes it into subtasks, assigns them to worker agents, and synthesizes the results.

**How it works:**
1. Orchestrator receives: "Write a market analysis report on electric vehicles"
2. Orchestrator decomposes into: research → analysis → writing → review
3. Each worker agent receives its subtask and returns results
4. Orchestrator combines all outputs into a final response

**Strengths:** Clear control flow, easy to debug, worker agents are simple.

**Weaknesses:** Orchestrator is a bottleneck and single point of failure. All communication routes through one agent, creating latency.

This is the pattern used by **CrewAI** (role-based crews with process types) and **OpenAI's Agents SDK** (explicit handoffs).

### 2.2 Pipeline (Sequential)

Agents are arranged in a chain. Each agent receives the previous agent's output, processes it, and passes the result to the next agent.

**How it works:**
1. Agent A (researcher): gathers raw information
2. Agent B (analyst): processes and structures the data
3. Agent C (writer): creates the final document

**Strengths:** Simple to implement, each agent's context is focused on its specific job.

**Weaknesses:** No parallelism — total latency is the sum of all agent times. Errors cascade: if Agent A hallucinates a fact, Agents B and C will faithfully propagate it.

This pattern is natural for content creation pipelines and is supported by all major frameworks.

### 2.3 Mesh (Peer-to-Peer)

All agents can communicate with all other agents. There is no fixed hierarchy.

**How it works:**
1. All agents share a common message space
2. Any agent can send a message to any other agent
3. Coordination emerges from the agents' own reasoning about who to talk to

**Strengths:** Highly flexible, resilient to individual agent failures, can handle unexpected task structures.

**Weaknesses:** Hard to debug, communication overhead scales quadratically with agent count, risk of infinite loops or "groupthink."

**AutoGen's GroupChat** mode uses this pattern — agents take turns in a shared conversation, and a built-in router (sometimes another LLM) decides who speaks next.

---

## 3. The Multi-Agent Lifecycle

![Figure 2: The lifecycle of a multi-agent task — from user request through orchestration, parallel execution, and synthesis](../zh/images/day36/multi-agent-lifecycle.png)
*Figure 2: A typical multi-agent task lifecycle. The orchestrator decomposes, workers execute in parallel, and an aggregator synthesizes the final response.*

Let's trace a concrete example through the full lifecycle. Suppose the user asks: "Compare the top three cloud providers for running LLM inference."

### 3.1 Task Decomposition

The orchestrator's first job is to decide *whether* to use multiple agents and, if so, *how* to split the work. This is harder than it sounds.

A good decomposition is:
- **Agent 1**: Research AWS offerings (SageMaker, Bedrock, EC2 GPU instances)
- **Agent 2**: Research Google Cloud offerings (Vertex AI, TPU access, Gemini API)
- **Agent 3**: Research Azure offerings (Azure AI, AKS with GPU)

A bad decomposition would be:
- **Agent 1**: "Do everything for AWS and Google"
- **Agent 2**: "Review what Agent 1 did"

The decomposition must create **independent subtasks** with **clear interfaces** — just like good software design.

### 3.2 Execution and Context Sharing

Each worker agent needs context to do its job. There are two strategies:

**Shared memory**: All agents read from and write to a common workspace (like a shared document or database). This is fast but can lead to conflicts if two agents modify the same section simultaneously.

**Message passing**: Agents explicitly send messages to each other. This is cleaner but adds latency. Each message is a structured payload containing the relevant context subset.

Most real systems use a hybrid: a shared workspace for persistent knowledge plus targeted messages for coordination.

### 3.3 Aggregation and Synthesis

The aggregator's job is to combine the worker outputs into a coherent final response. This is where many systems fail — simply concatenating three agent outputs does not produce a good report.

A good aggregator will:
1. **Deduplicate**: Remove overlapping information across agents
2. **Reconcile**: Resolve contradictions (Agent A says X is best, Agent B says Y is best)
3. **Structure**: Organize into a coherent narrative with consistent formatting
4. **Verify**: Check for hallucinations or unsupported claims

---

## 4. Failure Modes — Why Multi-Agent Systems Break

![Figure 3: Three common multi-agent failure modes — error cascade, coordination drift, and redundant work](../zh/images/day36/multi-agent-failure-modes.png)
*Figure 3: The most common failure modes in multi-agent LLM systems, based on the MAST failure taxonomy (Cemri et al., 2025).*

This is the section that matters most if you're building production multi-agent systems. Research consistently shows that **most multi-agent deployments fail within weeks** — not from coding errors, but from predictable coordination breakdowns.

### 4.1 Error Cascade

#### Intuition: The Assembly Line

Imagine a car factory where the first station installs the wrong engine. Every subsequent station — transmission, exhaust, electrical — will build on top of the wrong foundation. The final car won't work, but the problem started at station one.

In multi-agent systems, if Agent A (researcher) retrieves incorrect information, Agents B, C, and D will all build on that false foundation. The error compounds at each step.

**The MAST study** found that error cascades are the most common failure mode, accounting for roughly 30% of observed failures. The more agents in the pipeline, the worse the amplification.

**Mitigation**: Insert verification checkpoints between stages. A lightweight "fact-checker" agent can catch cascading errors early before they propagate.

### 4.2 Coordination Drift

#### Intuition: The Group Project

Remember group projects in school? The team starts aligned, but gradually each person interprets the requirements differently. By the deadline, three people have written overlapping sections and nobody covered the conclusion.

The 2026 "Agent Drift" study identified three drift types:

| Drift Type | What Happens | Example |
|------------|-------------|---------|
| **Semantic drift** | Gradual departure from original intent | Research agent starts summarizing instead of analyzing |
| **Coordination drift** | Consensus breakdown between agents | Writer and reviewer disagree on what "concise" means |
| **Behavioral drift** | Emergence of unintended strategies | Agents learn to agree with each other instead of criticizing |

**Mitigation**: Periodically re-inject the original task description into each agent's context. Use structured output formats (JSON schemas) to enforce consistent interfaces between agents.

### 4.3 Redundant Work and Groupthink

When agents don't have clear task boundaries, they often duplicate effort. Two agents independently research the same subtopic, wasting compute and time.

Worse, in mesh topologies, agents can fall into **groupthink** — a tendency where agents reinforce each other's errors rather than providing independent evaluation. Research by Wynn et al. (2025) found that debate-style multi-agent systems can suppress dissent, leading to confident but wrong consensus.

**Mitigation**: Assign agents explicit non-overlapping scopes. In review setups, give the reviewer agent a different system prompt that emphasizes skepticism and critical thinking.

### 4.4 The MAST Failure Taxonomy

The most comprehensive catalog of multi-agent failures comes from the MAST study (Cemri et al., NeurIPS 2025), which analyzed 1,600+ execution traces. The 14 failure modes map to three root categories:

| Category | Description | Key Failure Modes |
|----------|-------------|-------------------|
| **Specification ambiguity** | The task or agent roles are unclear | Vague instructions, missing constraints |
| **Coordination breakdown** | Agents fail to communicate effectively | Missed handoffs, conflicting outputs |
| **Verification gaps** | No mechanism to check intermediate results | Unchecked hallucinations, silent failures |

The practical takeaway: **invest in specification and verification before adding more agents**. A well-specified two-agent system outperforms a poorly specified five-agent system.

---

## 5. Frameworks and the 2026 Protocol Stack

![Figure 4: The 2026 enterprise agent protocol stack — A2A for agent-to-agent communication, MCP for tool access, and application frameworks](../zh/images/day36/agent-protocol-stack.png)
*Figure 4: The emerging protocol stack for multi-agent systems in 2026. MCP handles agent-to-tool communication, A2A handles agent-to-agent communication.*

The multi-agent ecosystem in 2026 is shaped by two major protocols and several competing frameworks.

### 5.1 MCP — The Tool Access Layer

**Model Context Protocol (MCP)**, introduced by Anthropic in November 2024 and donated to the Linux Foundation's Agentic AI Foundation in December 2025, standardizes how agents access external tools and data sources.

Think of MCP like USB for AI agents — a universal connector. Instead of each agent needing custom code to talk to each database, API, or file system, MCP provides a standard interface.

- **What it solves**: Agent-to-tool communication
- **Key capability**: Tool discovery, context sharing, data connections
- **Adoption**: Supported by OpenAI, Google DeepMind, Microsoft; millions of monthly SDK downloads

### 5.2 A2A — The Agent Communication Layer

**Agent-to-Agent Protocol (A2A)**, launched by Google in April 2025 and contributed to the Linux Foundation in June 2025, handles the harder problem: how agents discover and communicate with *other agents*.

- **What it solves**: Agent-to-agent communication
- **Key capability**: Agent discovery via "Agent Cards" (JSON capability descriptions), structured task lifecycle (pending → in-progress → completed → failed), real-time streaming via Server-Sent Events
- **Version**: Reached v1.2 by April 2026 with cryptographic signatures for agent verification
- **Adoption**: 150+ organizations including Salesforce, SAP, PayPal, ServiceNow

MCP and A2A are **complementary**, not competing:
- **MCP** = "How do I use this tool?" (agent → database, API, filesystem)
- **A2A** = "How do I talk to that agent?" (agent → agent)

### 5.3 Framework Comparison

| Framework | Topology | Best For | Key Feature |
|-----------|----------|----------|-------------|
| **[LangGraph](https://github.com/langchain-ai/langgraph)** | Directed graph | Production stateful systems | Battle-tested, conditional edges, 100+ LLM support |
| **[CrewAI](https://github.com/crewAIInc/crewAI)** | Role-based crews | Rapid prototyping | 10 lines → working crew, 40% faster time-to-production |
| **[AutoGen (AG2)](https://github.com/ag2ai/ag2)** | Conversational GroupChat | Research, flexible collaboration | Event-driven async, human-in-the-loop |
| **[Google ADK](https://github.com/google/adk-python)** | Hierarchical agent tree | Google Cloud integration | Native A2A, Vertex AI integration |
| **[OpenAgents](https://openagents.org)** | Open protocol mesh | Cross-framework interoperability | Native MCP + A2A support |

The right choice depends on your use case: CrewAI for speed, LangGraph for production reliability, AutoGen for research flexibility.

---

## 6. The Scaling Myth — More Agents ≠ Better Results

> This section provides the mathematical intuition behind why multi-agent systems don't scale linearly.

The communication overhead in a multi-agent system follows a well-known pattern from distributed systems theory. For **n** agents in a mesh topology, the number of potential communication channels is:

$$
\text{Channels} = \frac{n(n-1)}{2}
$$

This is quadratic growth. With 3 agents, you have 3 channels. With 7 agents, you have 21. With 10 agents, 45. Each channel is a potential source of miscommunication, conflicting information, or coordination overhead.

The effective quality of a multi-agent system can be approximated as:

$$
Q_{\text{system}} = \frac{n \times q_{\text{agent}}}{1 + \alpha \times \frac{n(n-1)}{2}}
$$

Where **q_agent** is individual agent quality, **n** is agent count, and **\alpha** is the coordination overhead per channel (typically 0.05-0.15 depending on task complexity). As **n** grows, the denominator grows quadratically while the numerator grows linearly — there is a clear optimum beyond which adding agents hurts.

Solving for the optimal agent count:

$$
n^* \approx \sqrt{\frac{2}{\alpha}}
$$

For typical values of **\alpha** (0.05-0.15), this gives an optimal team size of 3-6 agents — exactly matching what empirical research observes.

One of the most important findings from 2025-2026 research is that **adding more agents often makes systems worse**.

A February 2026 article in *Towards Data Science* documented the "17x Error Trap" — systems where naive multi-agent approaches produced error rates 17 times higher than single-agent baselines. The root cause: the "bag of agents" pattern, where agents are added without careful coordination architecture.

The research shows a non-monotonic relationship between agent count and performance:

- **1 agent**: Good for simple, well-defined tasks
- **2-3 agents**: Sweet spot for most real tasks (researcher + writer + reviewer)
- **4-6 agents**: Marginal improvements if carefully orchestrated
- **7+ agents**: Typically worse — coordination overhead exceeds benefits

The key variable is not *how many* agents, but the **topology of coordination**. Arranging agents into functional planes — planning, execution, verification — transforms a noisy bag of agents into a closed-loop system.

---

## 7. Code Example — Minimal Multi-Agent with Handoffs

Here's a simplified Python example showing the star topology pattern with explicit handoffs:

```python
import json
from typing import Any

class Agent:
    """A minimal agent that can process tasks and hand off results."""
    
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.context: list[dict] = []
    
    def process(self, task: str, shared_workspace: dict) -> str:
        """Process a task using the agent's specialized role."""
        # In practice, this would call an LLM with:
        # - system_prompt (defining the agent's role)
        # - task (current assignment)
        # - context (previous conversation)
        # - shared_workspace (information from other agents)
        
        prompt = f"""
        You are a {self.role}.
        {self.system_prompt}
        
        Task: {task}
        
        Shared context: {json.dumps(shared_workspace, indent=2)}
        """
        
        # Placeholder for LLM call (e.g., openai.chat.completions.create)
        result = f"[{self.name} output for: {task}]"
        
        # Write results back to shared workspace
        shared_workspace[self.name] = {
            "role": self.role,
            "output": result
        }
        
        return result


class Orchestrator:
    """Central coordinator for a multi-agent system (star topology)."""
    
    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self.workspace: dict[str, Any] = {}
    
    def register(self, agent: Agent):
        """Add a specialized agent to the team."""
        self.agents[agent.name] = agent
    
    def run(self, user_request: str) -> str:
        """Decompose task, run agents, synthesize results."""
        # Step 1: Decompose (in practice, use an LLM call)
        subtasks = {
            "researcher": f"Research: {user_request}",
            "analyst": f"Analyze the research findings",
            "writer": f"Write a summary based on analysis"
        }
        
        # Step 2: Execute sequentially (could be parallel)
        for agent_name, task in subtasks.items():
            agent = self.agents[agent_name]
            agent.process(task, self.workspace)
        
        # Step 3: Synthesize (in practice, use an LLM call)
        return self.workspace


# Usage
orchestrator = Orchestrator()
orchestrator.register(Agent("researcher", "research specialist",
    "Find factual, well-sourced information. Cite your sources."))
orchestrator.register(Agent("analyst", "data analyst",
    "Identify patterns, trade-offs, and key insights."))
orchestrator.register(Agent("writer", "technical writer",
    "Create clear, structured prose. Be concise."))

result = orchestrator.run("Compare LLM inference options on AWS vs GCP")
```

This is intentionally simplified. Real production systems add error handling, retries, parallel execution, and structured output parsing. But the core pattern — decompose, execute, synthesize — is the same.

---

## 8. Common Misconceptions

### ❌ "More agents = better results"

Adding agents increases coordination overhead exponentially. The sweet spot for most tasks is 2-4 agents with clear roles. Research consistently shows that beyond 5-6 agents, performance degrades unless you invest heavily in orchestration logic.

### ❌ "Multi-agent systems are self-organizing"

Unlike ant colonies or flocking birds, LLM agents don't naturally self-organize into efficient structures. They need explicit instructions about when to speak, what to produce, and how to coordinate. The "emergence" in multi-agent systems is usually **emergent failure**, not emergent intelligence.

### ❌ "Each agent should use a different LLM"

While it's possible to use different models for different agents (e.g., GPT-4 for reasoning, Claude for writing), this adds complexity. Start with the same model and differentiate through system prompts. Only introduce model heterogeneity when you have a clear reason — like the 2026 ICRC paper on iterative critique-and-routing with heterogeneous LLMs showing improved results through model diversity ([arXiv:2605.08686](https://arxiv.org/abs/2605.08686)).

---

## 9. Frontier — What's New in Multi-Agent Systems

The field is moving rapidly. Here are the most significant developments from the last six months:

**1. Reinforcement Learning for Multi-Agent Orchestration (May 2026)**
A new paradigm trains orchestrator agents using reinforcement learning on orchestration traces — learning *how to coordinate* rather than relying on hand-crafted rules. The approach uses a systematic multi-agent RFT (Reinforcement Fine-Tuning) paradigm with hierarchical GRPO decomposition for LLM teams. ([arXiv:2605.02801](https://arxiv.org/abs/2605.02801))

**2. Cattle Trade Benchmark for Strategic Agent Behavior (May 2026)**
Researchers introduced a multi-agent benchmark testing LLM capabilities in bluffing, bidding, and bargaining — moving beyond cooperative tasks to test strategic and competitive agent behavior. ([arXiv:2605.14537](https://arxiv.org/abs/2605.14537))

**3. Constitutional Design for Multi-Agent Governance (May 2026)**
New work explores how multi-agent systems should govern themselves. Rather than fixed human-authored rules, the research investigates how constitutions emerge from multi-agent interaction — treating alignment as a dynamic social process rather than a static rulebook. ([arXiv:2605.09128](https://arxiv.org/abs/2605.09128))

**4. MAST Failure Taxonomy (NeurIPS 2025)**
The most comprehensive empirical study of multi-agent failures to date, cataloging 14 failure modes across 1,600+ traces. This is the foundation for anyone building production multi-agent systems. ([Cemri et al., NeurIPS 2025](https://www.augmentcode.com/guides/why-multi-agent-llm-systems-fail-and-how-to-fix-them))

**5. A2A Protocol v1.2 (April 2026)**
Google's Agent-to-Agent protocol reached version 1.2, adding cryptographic signatures for agent verification and native integration with LangGraph, CrewAI, AutoGen, and Google ADK. The protocol is now supported by 150+ organizations. ([Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade))

---

## 10. Recommended Resources

## Appendix: The MAST Failure Taxonomy — 14 Multi-Agent Failure Modes Explained

In 2025, Cemri et al. from UC Berkeley analyzed 200 conversation traces (each averaging over 15,000 lines of text) across 7 popular multi-agent frameworks (MetaGPT, ChatDev, HyperAgent, OpenManus, AppWorld, Magentic, AG2), using Grounded Theory for systematic annotation, ultimately identifying **14 unique failure modes** clustered into 3 major categories. The percentages below are based on analysis of 1,642 execution traces.

> **What is Grounded Theory?** A qualitative research method introduced by Glaser and Strauss in 1967. The core idea is to **not presuppose answers, but build a classification system bottom-up from the data** — researchers read raw data line by line, label observed phenomena (open coding), constantly compare new observations with existing labels (constant comparison), and continue until no new categories emerge (theoretical saturation). For MAST, six experts used this approach to read agent conversation traces with "fresh eyes," marking failures as they found them and ultimately converging on 14 modes — rather than fitting data to a preconceived checklist. This method is particularly suited for "exploring unknown taxonomies."

> **Paper**: Cemri, M., Pan, M.Z., Yang, S. et al. "Why Do Multi-Agent LLM Systems Fail?" NeurIPS 2025 Datasets and Benchmarks Track Spotlight. [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)

---

### Category 1: System Design Issues (FC1)

Accounting for **~44%** of failures — the most common and fundamental category. The problem isn't agent capability but flawed system specification: vague task definitions, unclear role boundaries, missing termination conditions. Agents don't know what to do or when they're done.

| ID | Failure Mode | % | Description |
|------|---------|------|------|
| FM-1.1 | **Disobey task specification** | 11.8% | Agent fails to follow specified constraints or requirements, leading to suboptimal or incorrect outcomes. E.g., asked to generate Python 3 code but uses Python 2 syntax. |
| FM-1.2 | **Disobey role specification** | 1.5% | Agent oversteps its assigned role and behaves like another agent. E.g., a tester starts writing code, or a CEO agent usurps the CTO's responsibilities. |
| FM-1.3 | **Step repetition** | 15.7% | Agent unnecessarily repeats previously completed steps. This is the single most common failure mode — the agent "forgets" it already did something. |
| FM-1.4 | **Loss of conversation history** | 2.8% | Context is unexpectedly truncated, causing the agent to disregard recent interaction history and revert to an earlier conversational state. Common with context window overflow. |
| FM-1.5 | **Unaware of termination conditions** | 12.4% | Agent doesn't recognize that the task is complete and keeps running unnecessarily, or conversely, stops before conditions are met. |

**Why:** The root cause is typically the MAS designer's responsibility — system prompts that are too vague, incomplete workflow definitions, or no clear completion/stop signals. The paper argues that a well-designed MAS should interpret high-level objectives with minimal but precise user input.

---

### Category 2: Inter-Agent Misalignment (FC2)

Accounting for **~33%** of failures — even with perfect role definitions, communication and coordination between agents can still break down. Misread intentions, information gaps, and reasoning-action disconnects constitute the second largest failure category.

| ID | Failure Mode | % | Description |
|------|---------|------|------|
| FM-2.1 | **Conversation reset** | 2.2% | Agent unexpectedly restarts a dialogue, losing all previous context and progress. Often co-occurs with FM-1.4 (history loss). |
| FM-2.2 | **Fail to ask for clarification** | 6.8% | Agent proceeds based on wrong assumptions instead of requesting more information when faced with unclear or incomplete data. |
| FM-2.3 | **Task derailment** | 7.4% | Agent deviates from the original objective and starts working on irrelevant content. E.g., a research agent gets sidetracked by tangential information during a search. |
| FM-2.4 | **Information withholding** | 0.85% | Agent possesses critical information but fails to share it with other agents, causing uninformed decisions downstream. |
| FM-2.5 | **Ignored other agent's input** | 1.9% | Agent disregards or inadequately considers input or recommendations from other agents. E.g., a writer agent skips the reviewer's suggested edits. |
| FM-2.6 | **Reasoning-action mismatch** | 13.2% | Agent's reasoning process ("I should do X") diverges from its actual executed behavior (actually does Y). The agent "says one thing and does another." |

**How:** The paper emphasizes that diagnosing these failures is more complex than FC1. Similar surface behaviors (e.g., missing information) can stem from different root causes — is it FM-2.4 (withholding), FM-2.5 (ignoring input), or FM-1.4 (history loss)? This is why MAST needs fine-grained modes rather than a blanket "communication failure" label.

---

### Category 3: Task Verification (FC3)

Accounting for **~23%** of failures — even if the first two categories are solved, errors still slip through without effective verification mechanisms. The core question: **who checks the results, and how?**

| ID | Failure Mode | % | Description |
|------|---------|------|------|
| FM-3.1 | **Premature termination** | 6.2% | Ending a task before all necessary information has been exchanged or objectives have been met. E.g., returning "task complete" before the code is finished. |
| FM-3.2 | **No or incomplete verification** | 8.2% | System completely skips checking outputs, or performs only superficial validation. E.g., a ChatDev-generated chess program passes compilation but has severe runtime game-rule bugs. |
| FM-3.3 | **Incorrect verification** | 9.1% | The verification process itself is flawed, judging incorrect results as correct. The verifier agent makes its own mistakes, giving a false "looks good" verdict. |

**Why & How:** A key finding is that systems with explicit verifier agents (like MetaGPT, ChatDev) do have lower total failure rates — but verifier agents are no silver bullet. Existing verifiers often perform only superficial checks (does it compile? does it produce output?) despite being prompted to do thorough validation. This means verification itself needs to be designed more rigorously, not just "add a reviewer."

---

### Ranked by Frequency

From most to least common:

1. **FM-1.3 Step repetition** (15.7%) — most prevalent failure
2. **FM-1.1 Disobey task specification** (11.8%)
3. **FM-1.5 Unaware of termination conditions** (12.4%)
4. **FM-2.6 Reasoning-action mismatch** (13.2%)
5. **FM-3.3 Incorrect verification** (9.1%)
6. **FM-3.2 No or incomplete verification** (8.2%)
7. **FM-2.3 Task derailment** (7.4%)
8. **FM-3.1 Premature termination** (6.2%)
9. **FM-2.2 Fail to ask for clarification** (6.8%)
10. **FM-1.4 Loss of conversation history** (2.8%)
11. **FM-2.1 Conversation reset** (2.2%)
12. **FM-2.5 Ignored other agent's input** (1.9%)
13. **FM-1.2 Disobey role specification** (1.5%)
14. **FM-2.4 Information withholding** (0.85%) — rarest but most insidious

### Practical Implications

- **The top 5 modes account for ~62% of failures** — solving these first yields the largest reliability improvement
- **FC1 (system design) + FC2 (misalignment) combined = 79%** — meaning most failures are preventable through better specification engineering and coordination protocols
- No one-size-fits-all solution — different frameworks show very different failure profiles (e.g., OpenManus tends toward step repetition, HyperAgent toward reasoning-action mismatch)
- Fixes also vary: some require workflow redesign, others need stricter prompt engineering, some need independent verification mechanisms

---

## Further Reading

> The following resources are tiered by difficulty for readers at different stages.

### Beginner
1. [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-python) — Official guide to building multi-agent systems with OpenAI's SDK
2. [CrewAI Documentation](https://docs.crewai.com/) — The fastest way to build multi-agent teams; great for learning patterns

### Advanced
1. ["Why Your Multi-Agent System is Failing: Escaping the 17x Error Trap"](https://towardsdatascience.com/why-your-multi-agent-system-is-failing-escaping-the-17x-error-trap-of-the-bag-of-agents/) — Practical guide to coordination architecture
2. ["Multi-Agent Systems: From Classical Paradigms to Large Foundation Model-Enabled Futures"](https://arxiv.org/abs/2604.18133) — Comprehensive survey of multi-agent evolution

### Papers
1. ["Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces"](https://arxiv.org/abs/2605.02801) — RL-based orchestration (May 2026)
2. ["Cattle Trade: A Multi-Agent Benchmark for LLM Bluffing, Bidding, and Bargaining"](https://arxiv.org/abs/2605.14537) — Strategic agent behavior benchmark (May 2026)
3. ["Iterative Critique-and-Routing Controller for Multi-Agent Systems with Heterogeneous LLMs"](https://arxiv.org/abs/2605.08686) — Heterogeneous model coordination (May 2026)
4. ["Internal vs. External: Comparing Deliberation and Evolution for Multi-Agent Constitutional Design"](https://arxiv.org/abs/2605.09128) — Multi-agent governance (May 2026)
5. ["Agents of Chaos: LLM Agent Failures"](https://arxiv.org/abs/2602.20021) — Failure mode taxonomy (February 2026)

---

## Reflection Questions

1. If you were building a customer support system, would you use a single agent or a multi-agent architecture? What factors would determine your choice?
2. The MAST study found that specification ambiguity is the root cause of many failures. How would you design a "specification language" for multi-agent task decomposition?
3. A2A and MCP are complementary protocols, but what happens when they conflict — for example, when an agent discovers a tool via MCP that another agent is already using via A2A? How should the system resolve this?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| Coordination topology | How agents are connected — star, pipeline, or mesh |
| Task decomposition | Breaking a complex request into independent subtasks for agents |
| Error cascade | When one agent's mistake propagates through all downstream agents |
| Coordination drift | Agents gradually losing shared understanding over time |
| MCP (Model Context Protocol) | Standard for agent-to-tool communication |
| A2A (Agent-to-Agent Protocol) | Standard for agent-to-agent discovery and communication |
| MAST failure taxonomy | 14 failure modes across specification, coordination, and verification |
| Scaling myth | More agents often hurts — topology matters more than count |

**Key Takeaway**: Multi-agent systems are powerful but dangerous. The secret is not adding more agents — it's designing the right coordination topology, investing in specification clarity, and building verification checkpoints. Think of it as organizational design for AI: the org chart matters more than the headcount.

---

*Day 36 of 60 | LLM Fundamentals*
*Word count: ~2800 | Reading time: ~14 minutes*
