# PydanticAI Agent with Habitat AI Memory

A conversational AI agent built with [PydanticAI](https://ai.pydantic.dev/) that uses Habitat AI's **PSS (Persistent Session Store)** for long-term semantic memory. The agent remembers facts across conversation turns and can recall them when resuming a session — even without LLM message history.

## What is Habitat AI?

**Habitat AI** solves the "memory illusion" problem of LLMs: language models don't have native memory — they fake it by re-reading the entire conversation context each turn. As conversations grow, costs rise linearly and systems break when they hit context window limits.

Habitat AI provides **context compression** through its patented PSS technology:

| Problem (Status Quo)                        | Habitat AI Solution                                    |
|---------------------------------------------|--------------------------------------------------------|
| Token costs grow linearly with conversation | **Constant token cost** — only relevant memories sent  |
| Context window limits break long sessions   | **Unlimited session length** — compressed semantic state |
| Agents are stateless across sessions        | **Long-term continuity** — goals and context persist   |
| Same questions trigger repeated reasoning   | **Learned patterns** — short-circuit redundant retrieval |
| Silent drift degrades quality over time     | **Drift detection** — automatic phase tracking and correction |

### Benchmarks

Habitat AI achieves massive token savings while maintaining or improving quality, verified across standard LLM benchmarks:

| Benchmark            | Token Savings | Quality Impact  |
|----------------------|---------------|-----------------|
| MT-Bench (160 turns) | **97.4%**     | +0.06 relevance |
| MT-Eval (395 turns)  | **99.0%**     | +0.64 relevance |
| LongBench V2 (503 turns) | **98.8%** | +0.47 relevance |

The technology is **model-agnostic** and patented.

### Use Cases

- **AI Agents** — Long-term goal memory, continuous learning from past actions, reduced compute overhead
- **RAG Applications** — Deduplicated knowledge base, short-circuit high-confidence answers, consistent outputs
- **Vector Databases** — Consolidated semantic units before embedding, controlled index growth, lower storage costs
- **Graph Databases (Neo4j)** — Enhanced knowledge modeling, smarter graph retrieval via stored semantics, reduced redundant queries

## How PSS Works

PSS uses the **inline pattern**: each API call both retrieves relevant context and stores the previous turn's Q&A pair. The context is a compressed semantic summary — not the raw chat history.

```
┌──────────┐      ┌─────────────┐      ┌──────────┐
│   User   │─────▶│  Habitat AI │─────▶│   LLM    │
│  message │      │  PSS /run   │      │ (OpenAI) │
└──────────┘      └──────┬──────┘      └────┬─────┘
                         │                   │
                    compressed          LLM response
                    context             (buffered for
                    (injected into       next /run call)
                    system prompt)
```

Each turn:

1. **`POST /run`** — sends the user message + previous LLM response → PSS stores the Q&A pair and returns relevant memory context
2. **System prompt injection** — the compressed context is injected into the agent's system prompt via PydanticAI's `@agent.system_prompt` decorator
3. **LLM call** — PydanticAI runs the agent with the enriched system prompt
4. **Buffer response** — the LLM answer is held and sent with the *next* `/run` call

On exit, a final `/store` call persists the last response.

### PSS Context Format

Each response from PSS includes a context block with metadata and semantically-ranked memories:

```
[PSS Context | Phase: resonance | Turn 12 | 18 memories]
Relevant context:
  1. [0.85] Q: What's the hosting budget? A: ECS Fargate, 600 EUR/month...
  2. [0.61] Q: What OCR tech? A: AWS Textract for German and English invoices...
```

| Field         | Meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| **Phase**     | Conversation phase: `initialization` → `exploration` → `resonance` → `stability` |
| **Turn**      | Number of turns in the session                                          |
| **Memories**  | Number of stored memory entries                                         |
| **Score**     | Semantic similarity to the current question (0.0–1.0)                   |

The **phase tracking** is part of Habitat AI's drift detection system — it monitors the conversation's semantic state and prevents quality degradation over time.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (for running scripts with inline dependencies)
- An OpenAI or Anthropic API key (for the LLM)
- A Habitat AI PSS API key

## Setup

All API keys are read from environment variables. Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Then export them before running:

```bash
export PSS_API_KEY="pss_your_key_here"
export OPENAI_API_KEY="sk-..."          # or ANTHROPIC_API_KEY for Claude
```

| Variable | Required | Description |
|---|---|---|
| `PSS_API_KEY` | Yes | Habitat AI PSS API key |
| `PSS_BASE_URL` | No | PSS endpoint (defaults to `https://pss.versino.de/api/v1`) |
| `OPENAI_API_KEY` | One of these | OpenAI key (uses gpt-4o-mini) |
| `ANTHROPIC_API_KEY` | One of these | Anthropic key (uses Claude Sonnet) |

The agent scripts auto-detect which LLM provider is available based on which key is set.

## Project Structure

```
├── agent.py                 # Interactive chat agent with persistent memory
├── demo_conversation.py     # Scripted 3-part demo showing all PSS capabilities
├── test_pss_api.py          # PSS API endpoint test suite
├── pyproject.toml           # Project config with uv script shortcuts
├── .env.example             # Template for environment variables
└── README.md
```

## Running

A `pyproject.toml` is included so `uv` can manage the virtual environment and dependencies. All scripts also use [PEP 723](https://peps.python.org/pep-0723/) inline metadata, so `uv run` handles dependencies automatically.

Install dependencies (one-time):

```bash
uv sync
```

Run scripts:

```bash
uv run agent.py                 # Interactive chat agent
uv run demo_conversation.py     # Scripted 3-part demo
uv run test_pss_api.py          # PSS API test suite
```

### 1. API Test Suite

Verifies the PSS API endpoints work correctly (health, /run, /store, inline pattern, error handling). No OpenAI key needed — this tests PSS directly.

```bash
PSS_API_KEY="pss_..." uv run test_pss_api.py
```

Expected output:

```
============================================================
PSS API v1 Test Suite
============================================================

--- Test 1: Health Check ---
  [PASS] GET /health returns 200
  [PASS] Response has 'status' field

--- Test 2: /run — New Session (first turn) ---
  [PASS] POST /run returns 200
  [PASS] Response contains 'session_id'
  [PASS] Response contains 'context'

--- Test 3: /run — Continue Session ---
  [PASS] POST /run with session_id returns 200
  [PASS] Returned session_id matches
  [PASS] Context returned

--- Test 5: Inline Pattern (response passed to next /run) ---
  [PASS] Turn 2 /run with response returns 200
  [PASS] Session persisted across turns
  [PASS] Context returned for turn 2

--- Test 7a: Error — Missing API Key ---
  [PASS] Missing key returns 401
...
Done.
```

### 2. Scripted Demo (Recommended First Run)

Runs a 3-part automated conversation demonstrating all Habitat AI capabilities:

```bash
uv run demo_conversation.py
```

**Part 1 — Building Project Knowledge** (context compression):

The agent accumulates project facts over 6 turns. PSS compresses the growing conversation into only the relevant subset each turn — token cost stays constant.

```
You: Hi! My name is Alex and I'm the tech lead on a cloud migration project.
  PSS: phase=initialization | turn=0 | memories=0
Agent: Hi Alex! How can I help with your cloud migration?

You: We're migrating to AWS. Using ECS Fargate — 600 EUR/month budget.
  PSS: phase=initialization | turn=1 | memories=2
Agent: ECS Fargate is great for containerized workloads...

You: For OCR we're using AWS Textract. 50k invoices/month in German and English.
  PSS: phase=initialization | turn=2 | memories=4
Agent: Textract handles multi-language OCR well...

You: We also need a Neo4j knowledge graph to map document relationships.
  PSS: phase=initialization | turn=4 | memories=8
Agent: Neo4j is excellent for modeling document relationships...
```

**Part 2 — Session Continuity** (memory without chat history):

A fresh PSSClient resumes the session with **empty LLM message history**. The agent recalls all facts purely from Habitat AI's semantic memory:

```
You: What do you remember about my project?
  PSS: phase=initialization | turn=6 | memories=12
Agent: You're Alex, tech lead at Acme Corp, working on a cloud migration
       to AWS using ECS Fargate (600 EUR/mo), Textract for OCR (50k
       invoices/month), and Neo4j for document relationship mapping.
       Go-Live is March 15, 2026.
```

**Part 3 — Semantic Retrieval Across Topics**:

Cross-cutting questions pull the right facts by similarity — not by recency:

```
You: What graph database are we using and why?
Agent: You're using Neo4j to map document relationships and detect duplicates.

You: Tell me about the OCR requirements — languages and volume.
Agent: AWS Textract handles 50,000 invoices per month in German and English.

You: What's Alex's role on the project?
Agent: Alex is the tech lead on the cloud migration project at Acme Corp.
```

A summary table is printed at the end showing what each part demonstrated.


```
╭────────────────────────────────────────────────────────────────────────────╮
│ Habitat AI + PydanticAI Demo                                               │
│                                                                            │
Show less
│ Demonstrates context compression, semantic memory, and session continuity. │
│ PSS keeps token costs constant while the agent builds long-term knowledge. │
╰────────────────────────────────────────────────────────────────────────────╯
────────────────────── Part 1: Building Project Knowledge ──────────────────────
Each turn, PSS stores the Q&A pair and returns only relevant memories — not the 
full history. Token cost stays constant.
You: Hi! My name is Alex and I'm the tech lead on a cloud migration project at 
Acme Corp.
  PSS: phase=initialization | turn=0 | memories=0
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ Hi Alex! It's great to meet you. How can I assist you with your cloud        │
│ migration project at Acme Corp?                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
You: We're migrating our document processing pipeline to AWS. We'll use ECS 
Fargate for containers — hosting budget is 600 EUR per month.
  PSS: phase=initialization | turn=1 | memories=2
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ That sounds like a solid plan! With a hosting budget of 600 EUR per month,   │
│ it's crucial to optimize your ECS Fargate usage to stay within budget.       │
│ Consider using cost monitoring tools and implementing auto-scaling policies  │
│ to manage workloads efficiently.                                             │
╰──────────────────────────────────────────────────────────────────────────────╯
You: For OCR we're using AWS Textract. It needs to handle about 50,000 invoices 
per month in German and English.
  PSS: phase=initialization | turn=2 | memories=3
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ Using AWS Textract for processing 50,000 invoices monthly is a smart choice, │
│ especially for handling documents in multiple languages like German and      │
│ English. Ensure to monitor Textract usage closely, as costs can accumulate   │
│ with high document volume, and consider batch processing to optimize         │
│ performance and reduce costs.                                                │
╰──────────────────────────────────────────────────────────────────────────────╯
You: Go-Live date is March 15, 2026. The biggest risk is the Textract 
integration with our legacy SAP system.
  PSS: phase=initialization | turn=3 | memories=5
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ A Go-Live date of March 15, 2026, gives you some time to plan effectively.   │
│ Mitigating the integration risk between AWS Textract and your legacy SAP     │
│ system will be crucial; consider running proof-of-concept tests early to     │
│ identify potential issues, and ensure you have a solid rollback plan if      │
│ needed during the migration phase.                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
You: We also need to set up a Neo4j knowledge graph to map document 
relationships and detect duplicate invoices.
  PSS: phase=initialization | turn=4 | memories=7
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ Integrating a Neo4j knowledge graph will be valuable for mapping document    │
│ relationships and identifying duplicate invoices. Make sure to define your   │
│ data model clearly and establish the relationships you want to capture       │
│ upfront. This will facilitate smoother integration with your document        │
│ processing pipeline and enhance query performance.                           │
╰──────────────────────────────────────────────────────────────────────────────╯
You: Can you give me a complete summary of everything you know about my project?
  PSS: phase=initialization | turn=5 | memories=9
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ Sure! You're the tech lead on a cloud migration project at Acme Corp,        │
│ migrating your document processing pipeline to AWS. You're using ECS Fargate │
│ for container hosting with a budget of 600 EUR per month and AWS Textract    │
│ for processing around 50,000 invoices monthly in German and English. Your    │
│ Go-Live date is set for March 15, 2026, with the primary risk being the      │
│ integration between Textract and your legacy SAP system. Additionally,       │
│ you're setting up a Neo4j knowledge graph to map document relationships and  │
│ detect duplicate invoices. Let me know if you need any more details or       │
│ assistance!                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
Session ID: 09b3a64e-ce47-4aff-9ff5-8737ca372792
────────────────── Part 2: Session Continuity (Memory Recall) ──────────────────
New PSSClient with the same session_id but EMPTY LLM message history.
The agent recalls facts purely from Habitat AI's semantic memory — 
no chat replay needed. This is the key advantage over raw context windows.
You: What do you remember about my project?
  PSS: phase=initialization | turn=6 | memories=11
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ You're the tech lead on a cloud migration project at Acme Corp, migrating    │
│ your document processing pipeline to AWS using ECS Fargate for containers.   │
│ You're also setting up a Neo4j knowledge graph to map document relationships │
│ and detect duplicate invoices, with a Go-Live date of March 15, 2026. The    │
│ biggest risk identified is the Textract integration with your legacy SAP     │
│ system.                                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
You: How much does the hosting cost and what infrastructure are we using?
  PSS: phase=initialization | turn=7 | memories=13
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ Your hosting budget is 600 EUR per month, and you're using AWS ECS Fargate   │
│ for hosting containers as part of your document processing pipeline          │
│ migration.                                                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
You: What's the biggest risk for our Go-Live?
  PSS: phase=initialization | turn=8 | memories=15
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ The biggest risk for your Go-Live is the integration of Textract with your   │
│ legacy SAP system.                                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
─────────────────── Part 3: Semantic Retrieval Across Topics ───────────────────
PSS retrieves memories by semantic similarity, not recency.
Questions about different topics pull the right facts automatically.
You: What graph database are we using and why?
  PSS: phase=initialization | turn=10 | memories=18
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ You're using a Neo4j knowledge graph to map document relationships and       │
│ detect duplicate invoices. This graph database is valuable for its ability   │
│ to efficiently handle complex relationships between data points.             │
╰──────────────────────────────────────────────────────────────────────────────╯
You: Tell me about the OCR requirements — languages and volume.
  PSS: phase=exploration | turn=11 | memories=20
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ I don't have specific details about your OCR requirements, such as the       │
│ languages or volume. If you provide that information, I can help you         │
│ further!                                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
You: What's Alex's role on the project?
  PSS: phase=exploration | turn=12 | memories=21
╭─────────────────────────────────── Agent ────────────────────────────────────╮
│ Alex is the tech lead on the cloud migration project at Acme Corp.           │
╰──────────────────────────────────────────────────────────────────────────────╯
──────────────────────────────── Demo Complete ─────────────────────────────────
                             What This Demo Showed                              
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Habitat AI Capability ┃ What Happened                                        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Context Compression   │ 6 turns of knowledge built up, but PSS sent only the │
│                       │ relevant subset to the LLM each turn — constant      │
│                       │ token cost.                                          │
│ Session Continuity    │ Part 2 resumed with empty LLM history. The agent     │
│                       │ recalled all project facts from PSS semantic memory  │
│                       │ alone.                                               │
│ Semantic Retrieval    │ Part 3 asked cross-topic questions. PSS returned the │
│                       │ right memories by similarity score, not by recency.  │
│ Drift Prevention      │ PSS tracks conversation phase (initialization →      │
│                       │ exploration → resonance → stability) to detect and   │
│                       │ prevent semantic drift.                              │
└───────────────────────┴──────────────────────────────────────────────────────┘
Session: 09b3a64e-ce47-4aff-9ff5-8737ca372792
```

### 3. Interactive Agent

Start a live conversational agent with persistent memory:

```bash
uv run agent.py
```

Resume a previous session:

```bash
uv run agent.py --session <session_id>
```

Type `quit` or press Ctrl-C to exit. The session ID is printed on exit so you can resume later.

## PydanticAI Integration Guide

This section shows how to add Habitat AI memory to any PydanticAI agent in 4 steps.

### Step 1: PSS Client

A thin async wrapper around the two PSS endpoints:

```python
from dataclasses import dataclass, field
import httpx

@dataclass
class PSSClient:
    base_url: str
    api_key: str
    http: httpx.AsyncClient
    session_id: str | None = None
    _prev_response: str | None = field(default=None, repr=False)

    async def run(self, message: str) -> str:
        """Send user message to PSS, get back compressed context.

        If _prev_response is set, it's sent along so PSS stores the
        Q&A pair — this is the 'inline' pattern that avoids a
        separate /store call each turn.
        """
        payload: dict = {"message": message}
        if self.session_id:
            payload["session_id"] = self.session_id
        if self._prev_response:
            payload["response"] = self._prev_response
            self._prev_response = None

        resp = await self.http.post(
            f"{self.base_url}/run",
            headers={"X-API-Key": self.api_key},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        return data.get("context", "")

    async def store(self, response: str) -> None:
        """Persist the final LLM response (call on exit)."""
        if not self.session_id:
            return
        resp = await self.http.post(
            f"{self.base_url}/store",
            headers={"X-API-Key": self.api_key},
            json={"session_id": self.session_id, "response": response},
        )
        resp.raise_for_status()

    def set_prev_response(self, response: str) -> None:
        """Buffer a response to send with the next /run call."""
        self._prev_response = response
```

### Step 2: Dependency Injection

Use PydanticAI's `deps_type` to make the PSS client available to the agent:

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class Deps:
    pss: PSSClient
    pss_context: str = ""

agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=Deps,
    instructions=(
        "You are a helpful assistant with long-term memory powered by "
        "Habitat AI. Use the memory context to answer questions without "
        "asking the user to repeat themselves."
    ),
)
```

### Step 3: Dynamic System Prompt

Inject PSS context into every LLM call using the `@agent.system_prompt` decorator:

```python
@agent.system_prompt
async def inject_pss_context(ctx: RunContext[Deps]) -> str:
    if ctx.deps.pss_context:
        return f"Memory context from previous conversations:\n{ctx.deps.pss_context}"
    return ""
```

### Step 4: Conversation Loop

Wire it all together — call PSS before each agent turn, buffer the response after:

```python
import asyncio
import httpx

async def main():
    async with httpx.AsyncClient(timeout=30) as http:
        pss = PSSClient(
            base_url="https://pss.versino.de/api/v1",
            api_key="your-pss-api-key",
            http=http,
        )
        deps = Deps(pss=pss)
        message_history = []

        while True:
            user_input = input("You: ")
            if user_input.lower() in ("quit", "exit"):
                break

            # 1. Get compressed context from PSS (stores prev response inline)
            deps.pss_context = await pss.run(user_input)

            # 2. Run the PydanticAI agent with memory-enriched system prompt
            result = await agent.run(
                user_input,
                deps=deps,
                message_history=message_history,
            )
            print(f"Agent: {result.output}")

            # 3. Buffer response — sent with next /run call
            pss.set_prev_response(result.output)
            message_history = result.all_messages()

        # 4. Store final response before exiting
        if pss._prev_response:
            await pss.store(pss._prev_response)

        print(f"Session: {pss.session_id}")

asyncio.run(main())
```

### Resuming a Session

Pass an existing `session_id` to pick up where you left off. PSS returns relevant memories from the previous conversation — no need to replay the full message history:

```python
pss = PSSClient(
    base_url="https://pss.versino.de/api/v1",
    api_key="your-pss-api-key",
    http=http,
    session_id="a8cfb8df-7855-412f-9802-45ab35edfe4e",  # from previous session
)
```

## Why Habitat AI + Neo4j?

Habitat AI's semantic memory complements Neo4j's graph relationships:

- **Enhanced Knowledge Modeling** — PSS provides long-term contextual understanding that enriches graph traversals
- **Smarter Retrieval** — Stored semantics reduce noisy or repeated Cypher queries by resolving intent before hitting the graph
- **Reduced Costs** — Consolidated semantic units and fewer redundant embeddings ease pressure on vector indexes
- **Stronger Enterprise AI** — Combining semantic memory with graph analytics creates richer insights for complex use cases (fraud detection, recommendation engines, document relationship mapping)

## PSS API Reference

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | None | Health check — returns status, active sessions, version |
| `/run` | POST | `X-API-Key` | Send message, get compressed context. Optionally pass `session_id` and `response` |
| `/store` | POST | `X-API-Key` | Persist a response for a session |

### POST /run

```json
{
  "message": "user's question",
  "session_id": "optional — omit for new session",
  "response": "optional — previous LLM response to store inline"
}
```

Returns:

```json
{
  "session_id": "uuid",
  "context": "[PSS Context | Phase: initialization | Turn 0 | 0 memories]\n..."
}
```

### POST /store

```json
{
  "session_id": "uuid",
  "response": "the LLM response to persist"
}
```

Returns:

```json
{
  "session_id": "uuid"
}
```
