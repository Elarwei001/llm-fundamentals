# Day 38: MCP (Model Context Protocol) — The USB-C Standard for AI

> **Core Question**: How do you connect every AI assistant to every data source and tool without writing N×M custom integrations?

---

## Opening

Imagine if every USB device needed a unique plug shape for every laptop. Your mouse would have one plug for a Dell, another for a MacBook, and a third for a ThinkPad. That was the world of USB before USB-C came along and said: *one connector, every device*.

That's exactly the problem AI faced before November 2024. Every AI assistant (Claude, ChatGPT, Gemini) had its own proprietary way to connect to external tools and data sources. If you wanted your AI to read GitHub, query a database, and search Slack, you had to write three separate integrations — and then rewrite them when you switched from Claude to ChatGPT.

The **Model Context Protocol (MCP)** is the USB-C of AI. It's an open standard, introduced by Anthropic in November 2024, that defines *one* way for any AI application to connect to any external system. Within 18 months, it went from an experimental protocol to an industry standard backed by Anthropic, OpenAI, Google, and Microsoft — with over 97 million monthly SDK downloads by May 2026.

In this article, we'll unpack how MCP works under the hood, why it caught on so fast, and what it means for anyone building with AI.

---

## 1. The Problem MCP Solves

### Intuition: The Power Strip Analogy

Think of MCP like a universal power strip. Before power strips, if you had 5 devices and 2 wall outlets, you'd need to keep plugging and unplugging. A power strip gives you one standardized interface — plug anything in, it just works.

Before MCP, the AI ecosystem had an **N × M integration problem**:

- **N** AI assistants (Claude, ChatGPT, Gemini, Copilot, etc.)
- **M** data sources and tools (GitHub, Slack, databases, file systems, APIs, etc.)
- Each AI assistant needed a **custom integration** for each data source
- That's **N × M** total integrations, and every new AI or data source multiplied the work

MCP reduces this to **N + M**: each AI assistant implements one MCP client, and each data source implements one MCP server. The protocol is the standardized interface between them.

![Figure 1: Before MCP, every AI needed custom integrations for every data source (N×M). After MCP, each side implements one standard interface (N+M).](../zh/images/day38/mcp-vs-traditional-api.png)
*Figure 1: MCP eliminates the N×M integration problem by providing a standard protocol between AI assistants and data sources.*

### Why Not Just Use REST APIs?

Good question. REST APIs are great for server-to-server communication, but they have three problems in the AI context:

1. **No discovery mechanism**: An AI assistant has no standard way to ask "what can you do?" — each API documents its endpoints differently
2. **No semantic context**: REST doesn't carry information about *what* a tool does in a way an LLM can understand
3. **No unified auth**: Each API has its own authentication, rate limiting, and error handling

MCP addresses all three: it has a built-in capability discovery mechanism, tools are described with natural-language annotations that LLMs can parse, and the latest spec includes standardized OAuth 2.0 flows.

---

## 2. MCP Architecture: How It Actually Works

### Intuition: The Restaurant Analogy

Imagine you're at a restaurant. You (the **Host**) don't go into the kitchen yourself. You tell the waiter (**Client**) what you want, the waiter passes your order to the kitchen (**Server**) using a standard order format (**JSON-RPC**), and the kitchen sends your food back through the waiter.

The key insight: **you never talk to the kitchen directly**. The protocol (the order format) is what makes it possible for any customer to order from any kitchen, without either side knowing the other's internal setup.

### Core Components

MCP has four core components:

| Component | Role | Example |
|-----------|------|---------|
| **MCP Host** | The AI application that the user interacts with | Claude Desktop, VS Code with Copilot, ChatGPT |
| **MCP Client** | Lives inside the Host; manages communication with servers | Protocol handler within the host app |
| **MCP Server** | Provides tools, resources, and prompts to the AI | GitHub MCP Server, File System MCP Server |
| **Transport** | The communication layer between Client and Server | stdio (local) or Streamable HTTP (remote) |

![Figure 2: MCP architecture — the Host contains the LLM and MCP Client, which communicates with MCP Servers via JSON-RPC over stdio or Streamable HTTP.](../zh/images/day38/mcp-architecture-overview.png)
*Figure 2: The MCP architecture. The Host application contains both the LLM and the MCP Client, which communicates with one or more MCP Servers through a standardized transport layer.*

### The Three Primitives

Every MCP Server can expose three types of capabilities:

**1. Tools** — Functions the AI can call to perform actions
- Example: `create_issue`, `search_code`, `send_email`
- Think of tools as *verbs* — things the AI can *do*

**2. Resources** — Data the AI can read
- Example: a file's contents, a database row, a GitHub issue
- Think of resources as *nouns* — things the AI can *read*

**3. Prompts** — Reusable prompt templates
- Example: "Summarize this codebase" with pre-configured context
- Think of prompts as *recipes* — pre-packaged instructions

### Transport Layer: stdio vs Streamable HTTP

MCP defines two standard transport mechanisms:

| Transport | Use Case | How It Works |
|-----------|----------|--------------|
| **stdio** | Local development, desktop apps | Client launches server as a subprocess; messages pass through stdin/stdout |
| **Streamable HTTP** | Remote/production deployments | Client sends HTTP POST requests; server can stream responses via Server-Sent Events (SSE) |

#### Intuition: stdio vs HTTP

Think of **stdio** like talking to someone in the same room — fast, direct, no network needed. **Streamable HTTP** is like a phone call — works across distances, but needs a dial tone (HTTP) and can handle long pauses (SSE streaming).

The stdio transport is simpler: the client spawns the MCP server process, writes JSON-RPC messages to its stdin, and reads responses from stdout. This is great for local tools.

Streamable HTTP is more powerful: the server runs as an independent HTTP service. The client sends POST requests containing JSON-RPC messages, and the server can respond either with a single JSON response or stream multiple messages back using SSE. This enables remote MCP servers, load balancing, and production deployments.

---

## 3. A Concrete Example: Tool Call Flow

Let's trace what happens when a user asks "What's my latest GitHub PR?"

![Figure 3: The full MCP tool call sequence — from user request through JSON-RPC message exchange to final response.](../zh/images/day38/mcp-request-response-flow.png)
*Figure 3: Step-by-step flow of an MCP tool call, showing how the user's natural language request gets translated into a structured JSON-RPC call and back.*

Here's what happens under the hood:

**Step 1 — Capability Discovery**

When the MCP Client first connects to a server, it sends an `initialize` request:

```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "capabilities": {},
    "clientInfo": {"name": "claude-desktop", "version": "1.0"}
  }
}
```

The server responds with its capabilities — which tools, resources, and prompts it offers. This is the **discovery** step that makes MCP self-describing.

**Step 2 — Tool List**

The client asks for available tools:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

The server responds with a list of tools, each described with a name, description, and JSON Schema for its input parameters. This description is designed to be **readable by LLMs** — the model uses it to decide which tool to call and how.

**Step 3 — Tool Invocation**

When the LLM decides it needs GitHub data, the client sends:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 2,
  "params": {
    "name": "list_pull_requests",
    "arguments": {
      "state": "open",
      "sort": "updated",
      "direction": "desc",
      "per_page": 1
    }
  }
}
```

**Step 4 — Response**

The server executes the tool and returns structured results:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "PR #42: Fix authentication token refresh bug (updated 2 min ago)"
      }
    ]
  }
}
```

The LLM then incorporates this result into its response to the user.

---

## 4. The Protocol Specification: Key Technical Details

### JSON-RPC 2.0 Foundation

MCP builds on **JSON-RPC 2.0**, a well-established remote procedure call protocol. This is a deliberate design choice — by using a known standard, MCP avoids reinventing the wheel for message encoding, error handling, and request/response correlation.

Every MCP message is a valid JSON-RPC message with three possible types:
- **Requests**: Have an `id`, expect a response
- **Notifications**: No `id`, fire-and-forget
- **Responses**: Reply to a specific request (contains the matching `id`)

The integration complexity reduction can be expressed simply:

$$
\text{Before MCP: } N \times M \text{ integrations} \quad \rightarrow \quad \text{After MCP: } N + M \text{ integrations}
$$

For 4 AI assistants and 10 data sources, that's 40 custom integrations reduced to 14 — a 65% reduction that grows more dramatic as the ecosystem expands.

### Protocol Versioning

MCP uses date-based versioning (e.g., `2024-11-05`, `2025-11-25`). The current stable spec is **2025-11-25**, released on MCP's first anniversary. A release candidate for **2026-07-28** was announced in May 2026, featuring stateless operation and an extensions framework.

#### Intuition: Date-Based Versioning

Think of it like software update dates on your phone. Instead of "v2.0" or "v3.0", MCP says "the November 2025 version" — it's unambiguous and tells you exactly when the spec was finalized.

### Key Features in the 2025-11-25 Spec

The November 2025 release was a major milestone. Here are the most important additions:

| Feature | What It Does | Why It Matters |
|---------|-------------|----------------|
| **Streamable HTTP transport** | Replaced the old HTTP+SSE with a cleaner design | Production deployments, load balancing |
| **Asynchronous Tasks** | Servers can run long operations without blocking | Data analysis, code generation |
| **OAuth 2.0 authorization** | Standard auth flow for remote servers | Enterprise security, multi-tenant access |
| **Server identity** | Servers can identify themselves | Trust, auditing |
| **Extensions framework** | Official mechanism for extending the protocol | Innovation without fragmentation |
| **Decoupled schemas** | Request payloads separated from RPC method definitions | Easier SDK generation |

### The Extensions Framework

One of the most forward-looking additions is the **extensions** system. Rather than baking every feature into the core spec, MCP now has a formal mechanism for extensions — optional capabilities that servers and clients can negotiate.

The first official extension is **MCP Apps** (launched January 2026), which enables MCP servers to deliver interactive UI components — dashboards, forms, data visualizations — directly inside the chat window. This means an MCP server can not only return text data but also render a live chart or a clickable form.

---

## 5. Security Considerations

MCP's rapid adoption has outpaced security best practices. In May 2026, the **NSA published official security guidance** for MCP deployments — a clear signal that the protocol has reached enterprise-critical status.

### Key Security Risks

| Risk | Description | Mitigation |
|------|------------|------------|
| **Tool poisoning** | Malicious MCP server provides harmful tool descriptions | Verify server identity; use allowlists |
| **Prompt injection via tools** | Tool results contain instructions that manipulate the LLM | Sanitize tool outputs; use content boundaries |
| **OAuth redirect theft** | Attacker intercepts authorization codes during OAuth flow | Use PKCE; validate redirect URIs strictly |
| **DNS rebinding** | Remote websites interact with local MCP servers | Validate `Origin` headers; bind to localhost only |
| **Privilege escalation** | Tool gains more access than intended | Principle of least privilege; scope tools narrowly |

#### Intuition: The Malicious Bartender Problem

Think of MCP security like hiring a bartender (MCP Server) for your restaurant (Host). If you hire without vetting, the bartender could slip something into the drinks (tool poisoning). Or a customer could hand the bartender a forged order (prompt injection). The solution: verify who you hire (server identity), check every order (input validation), and limit what the bartender can access (principle of least privilege).

---

## 6. Ecosystem and Adoption

### From Experiment to Industry Standard

![Figure 4: MCP SDK monthly downloads growth, from launch through May 2026, showing key adoption milestones.](../zh/images/day38/mcp-sdk-downloads-growth.png)
*Figure 4: MCP SDK downloads grew from 100K at launch to 97M+ monthly by May 2026, driven by adoption from every major AI provider.*

The adoption curve has been extraordinary:

| Milestone | Date | Significance |
|-----------|------|-------------|
| MCP launched | Nov 2024 | Anthropic open-sources the protocol |
| OpenAI adds MCP | Sep 2025 | Biggest competitor adopts the standard |
| MCP Registry launched | Sep 2025 | Central index for discovering MCP servers |
| Spec 2025-11-25 | Nov 2025 | Major spec update with production features |
| MCP Apps released | Jan 2026 | Interactive UI as an official extension |
| Donated to Linux Foundation | Mar 2026 | AAIF formed with Anthropic, Block, OpenAI |
| NSA security guidance | May 2026 | Government recognition of enterprise importance |
| Spec 2026-07-28 RC | Jul 2026 | Stateless operation, extensions framework |

### Who's Using MCP?

By mid-2026, MCP is embedded in production across the AI industry:

- **ChatGPT** (OpenAI): MCP support for third-party tool access since September 2025
- **Claude Desktop** (Anthropic): Over 75 connectors powered by MCP
- **VS Code / Copilot** (Microsoft): MCP for code intelligence and project context
- **Gemini** (Google): MCP integration for tool connectivity
- **Replit, Cursor, Sourcegraph**: AI coding assistants with MCP-powered context
- **Enterprise**: Databricks, Stripe, Notion, Bloomberg — all shipping MCP servers

### MCP Registry

The [MCP Registry](https://github.com/modelcontextprotocol/registry) serves as the central index for discovering available MCP servers. Launched in September 2025 with an initial batch, it grew by 407% to nearly 2,000 entries within months. Major entries include:

- [GitHub MCP Server](https://github.com/github/github-mcp-server) — Automate engineering workflows
- [Stripe MCP Server](https://docs.stripe.com/mcp) — Payment processing
- [Notion MCP Server](https://github.com/makenotion/notion-mcp-server) — Note management
- [Hugging Face MCP Server](https://github.com/huggingface/hf-mcp-server) — Model and dataset search

---

## 7. Building Your First MCP Server

Let's look at how simple it is to create an MCP server using the official Python SDK:

```python
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("my-tools")

# Define a tool — this becomes discoverable by any MCP client
@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The city name, e.g. "Singapore"
    
    Returns:
        A weather summary string
    """
    # In practice, you'd call a real weather API here
    return f"Weather in {city}: 28°C, partly cloudy"

# Define a resource — read-only data
@mcp.resource("config://app-settings")
def get_settings() -> str:
    """Return application settings as a resource."""
    return '{"theme": "dark", "language": "en"}'

# Define a prompt template
@mcp.prompt()
def summarize_code(code: str) -> str:
    """Generate a prompt for code summarization."""
    return f"Please summarize the following code and explain its purpose:\n\n{code}"

# Run the server using stdio transport (for local development)
if __name__ == "__main__":
    mcp.run()
```

This is remarkably concise. With just a few decorators, you've created an MCP server that any MCP-compatible AI assistant can discover and use. The `FastMCP` class handles JSON-RPC message routing, capability advertisement, and transport negotiation automatically.

---

## 8. Common Misconceptions

### ❌ "MCP is just another API framework"

No. MCP is a **protocol**, not a framework. The difference matters: a protocol defines *how* two parties communicate, while a framework defines *how you write code*. MCP doesn't care if you build your server in Python, TypeScript, Rust, or Go — it only cares that messages follow the JSON-RPC 2.0 format with MCP-specific methods.

### ❌ "MCP is only for Anthropic/Claude"

MCP was created by Anthropic, but it's now governed by the **Agentic AI Foundation (AAIF)** under the Linux Foundation, co-founded by Anthropic, Block, and OpenAI with support from Google, Microsoft, AWS, Cloudflare, and Bloomberg. Every major AI provider has adopted it.

### ❌ "MCP replaces REST APIs"

MCP complements REST, it doesn't replace it. MCP servers often wrap REST APIs internally — the MCP server acts as a translation layer that makes REST endpoints discoverable and understandable to LLMs. Think of MCP as the *interface* and REST as one possible *implementation*.

---

## 9. The 2026 Roadmap: What's Next

The MCP community has outlined four priority areas for 2026:

| Priority | Goal | Status |
|----------|------|--------|
| **Transport Evolution** | Stateless operation, horizontal scaling, `.well-known` discovery | In progress (2026-07-28 RC) |
| **Agent Communication** | Lifecycle management for Tasks (retry, expiry) | Experimental |
| **Governance Maturation** | Delegate SEP review to Working Groups | Active |
| **Enterprise Readiness** | Audit trails, SSO, gateway behavior, config portability | Community-driven |

The upcoming 2026-07-28 spec release candidate is particularly significant: it makes MCP **stateless**, meaning servers can run on commodity HTTP infrastructure without maintaining session state — a requirement for large-scale production deployments.

---

## 10. Further Reading

### Official Resources
1. [MCP Specification (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25) — The canonical spec document
2. [MCP Documentation](https://modelcontextprotocol.io/) — Official docs, tutorials, and SDK references
3. [MCP GitHub Repository](https://github.com/modelcontextprotocol/modelcontextprotocol) — Spec, issues, and contributions

### Blog Posts & Deep Dives
4. ["One Year of MCP: November 2025 Spec Release"](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — Official retrospective and spec changelog
5. ["The 2026 MCP Roadmap"](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — Priority areas and Working Group structure
6. ["Why the Model Context Protocol Won" — The New Stack](https://thenewstack.io/why-the-model-context-protocol-won/) — Industry analysis of MCP's success

### Security
7. [NSA MCP Security Guidance (May 2026)](https://www.nsa.gov/Portals/75/documents/Cybersecurity/CSI_MCP_SECURITY.pdf) — Official government security recommendations
8. [MCP Security Best Practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices) — Community-maintained security guide

### Key Announcements
9. [Anthropic donates MCP to Linux Foundation](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) — Formation of the AAIF
10. [MCP Apps Official Release (Jan 2026)](https://modelcontextprotocol.info/blog/mcp-apps-ui-capabilities/) — Interactive UI extension

---

## Reflection Questions

1. **Why did MCP succeed as a standard when many AI standardization attempts have failed?** Consider the role of timing, simplicity, and competitive dynamics.
2. **If MCP makes it trivial for any AI to connect to any tool, what new security risks emerge at scale?** Think about the "one standard = one attack surface" problem.
3. **How does MCP's extensions framework balance innovation speed against fragmentation risk?** Compare to how web standards evolved (CSS prefixes, WebGL extensions).

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| **MCP** | An open protocol standardizing how AI applications connect to external tools and data |
| **MCP Host** | The AI application (e.g., Claude Desktop, VS Code) that the user interacts with |
| **MCP Client** | The protocol handler inside the Host that manages server connections |
| **MCP Server** | A service that exposes tools, resources, and prompts to AI applications |
| **Tools** | Callable functions — things the AI can *do* (like creating a GitHub issue) |
| **Resources** | Readable data — things the AI can *read* (like a file's contents) |
| **Prompts** | Reusable prompt templates — pre-packaged instructions |
| **stdio transport** | Local communication via stdin/stdout (client spawns server as subprocess) |
| **Streamable HTTP** | Remote communication via HTTP POST + SSE streaming |
| **JSON-RPC 2.0** | The message encoding format MCP builds on |
| **MCP Registry** | The central index for discovering available MCP servers |
| **AAIF** | Agentic AI Foundation — the Linux Foundation body governing MCP |
| **MCP Apps** | An official extension enabling interactive UI inside chat windows |
| **SEPs** | Specification Enhancement Proposals — how the community evolves MCP |

**Key Takeaway**: MCP solved the AI industry's N×M integration problem with a simple, elegant protocol built on JSON-RPC 2.0. Its rapid adoption — from launch to 97M+ monthly SDK downloads in 18 months, with every major AI provider on board — makes it the de facto standard for connecting AI to the real world. Understanding MCP is now essential for anyone building AI-powered applications.

---

*Day 38 of 60 | LLM Fundamentals*
*Word count: ~2800 | Reading time: ~14 minutes*
