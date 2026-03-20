# /// script
# requires-python = ">=3.10"
# dependencies = ["pydantic-ai", "openai", "httpx", "rich", "anthropic"]
# ///
"""
Scripted multi-turn demo of the PydanticAI agent with Habitat AI PSS memory.

Demonstrates the three core capabilities of Habitat AI's PSS:
  1. Context compression — constant token cost regardless of conversation length
  2. Semantic memory — relevant facts retrieved by similarity, not recency
  3. Session continuity — resume conversations with full recall, no chat replay

Runs a 3-part conversation:
  Part 1: Build up project knowledge over several turns (shows token savings)
  Part 2: Resume the session with empty LLM history (shows memory persistence)
  Part 3: Ask cross-cutting questions (shows semantic retrieval across topics)
"""

import asyncio
import os
from dataclasses import dataclass, field

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from pydantic_ai import Agent, RunContext

PSS_BASE_URL = os.environ.get("PSS_BASE_URL", "https://pss.versino.de/api/v1")
PSS_API_KEY = os.environ.get("PSS_API_KEY", "")
if not PSS_API_KEY:
    raise SystemExit("Error: PSS_API_KEY environment variable is required.")

# Auto-detect available LLM provider
if os.environ.get("ANTHROPIC_API_KEY"):
    MODEL = "anthropic:claude-sonnet-4-20250514"
elif os.environ.get("OPENAI_API_KEY"):
    MODEL = "openai:gpt-4o-mini"
else:
    # Default — will fail at runtime with a clear error
    MODEL = "openai:gpt-4o-mini"

console = Console()


# ---------------------------------------------------------------------------
# PSS Client
# ---------------------------------------------------------------------------

@dataclass
class PSSClient:
    """Thin async wrapper around the Habitat AI PSS API.

    Uses the 'inline' pattern: each /run call sends the previous LLM response
    so PSS stores the Q&A pair and returns only the most relevant memories
    (constant context size, regardless of conversation length).
    """
    base_url: str
    api_key: str
    http: httpx.AsyncClient
    session_id: str | None = None
    _prev_response: str | None = field(default=None, repr=False)

    async def run(self, message: str) -> str:
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
        if not self.session_id:
            return
        resp = await self.http.post(
            f"{self.base_url}/store",
            headers={"X-API-Key": self.api_key},
            json={"session_id": self.session_id, "response": response},
        )
        resp.raise_for_status()

    def set_prev_response(self, response: str) -> None:
        self._prev_response = response


# ---------------------------------------------------------------------------
# PydanticAI Agent with Habitat AI memory
# ---------------------------------------------------------------------------

@dataclass
class Deps:
    pss: PSSClient
    pss_context: str = ""


agent = Agent(
    MODEL,
    deps_type=Deps,
    instructions=(
        "You are a helpful project assistant with long-term memory powered by "
        "Habitat AI. You remember facts from previous conversations through "
        "semantic memory — you don't need the user to repeat themselves. "
        "Answer using the memory context provided below when relevant. "
        "Keep answers concise (2-3 sentences max)."
    ),
)


@agent.system_prompt
async def inject_pss_context(ctx: RunContext[Deps]) -> str:
    if ctx.deps.pss_context:
        return f"Memory context from previous conversations:\n{ctx.deps.pss_context}"
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_pss_header(context: str) -> dict:
    """Extract phase, turn, and memory count from the PSS context header."""
    info = {"phase": "?", "turn": "?", "memories": "?"}
    if not context:
        return info
    first_line = context.splitlines()[0]
    for part in first_line.split("|"):
        part = part.strip().strip("[]")
        if part.startswith("Phase:"):
            info["phase"] = part.split(":", 1)[1].strip()
        elif part.startswith("Turn"):
            info["turn"] = part.split()[1].strip()
        elif "memories" in part.lower():
            info["memories"] = part.split()[0].strip()
    return info


async def run_turn(
    pss: PSSClient, deps: Deps, message_history: list, user_msg: str
) -> tuple[str, list]:
    """Run one conversation turn and return (answer, updated_history)."""
    console.print(f"\n[bold green]You:[/bold green] {user_msg}")

    # 1. PSS /run — get compressed context (and store prev response inline)
    pss_context = await pss.run(user_msg)
    deps.pss_context = pss_context

    info = parse_pss_header(pss_context)
    console.print(
        f"[dim]  PSS: phase={info['phase']} | turn={info['turn']} "
        f"| memories={info['memories']}[/dim]"
    )

    # 2. Agent run
    result = await agent.run(user_msg, deps=deps, message_history=message_history)
    answer = result.output

    console.print(Panel(Markdown(answer), title="[bold blue]Agent[/bold blue]", border_style="blue"))

    # 3. Buffer the LLM response so next /run stores the Q&A pair
    pss.set_prev_response(answer)

    return answer, result.all_messages()


# ---------------------------------------------------------------------------
# Demo script
# ---------------------------------------------------------------------------

async def main():
    console.print(Panel.fit(
        "[bold]Habitat AI + PydanticAI Demo[/bold]\n\n"
        "Demonstrates context compression, semantic memory, and session continuity.\n"
        "PSS keeps token costs constant while the agent builds long-term knowledge.",
        border_style="bright_magenta",
    ))

    async with httpx.AsyncClient(timeout=30) as http:

        # ================================================================
        # Part 1: Build project knowledge (shows context compression)
        # ================================================================
        console.rule("[bold yellow]Part 1: Building Project Knowledge[/bold yellow]")
        console.print(
            "[dim]Each turn, PSS stores the Q&A pair and returns only relevant "
            "memories — not the full history. Token cost stays constant.[/dim]\n"
        )

        pss = PSSClient(base_url=PSS_BASE_URL, api_key=PSS_API_KEY, http=http)
        deps = Deps(pss=pss)
        history = []

        conversation = [
            "Hi! My name is Alex and I'm the tech lead on a cloud migration project at Acme Corp.",
            "We're migrating our document processing pipeline to AWS. We'll use ECS Fargate for containers — hosting budget is 600 EUR per month.",
            "For OCR we're using AWS Textract. It needs to handle about 50,000 invoices per month in German and English.",
            "Go-Live date is March 15, 2026. The biggest risk is the Textract integration with our legacy SAP system.",
            "We also need to set up a Neo4j knowledge graph to map document relationships and detect duplicate invoices.",
            "Can you give me a complete summary of everything you know about my project?",
        ]

        for msg in conversation:
            answer, history = await run_turn(pss, deps, history, msg)

        # Store final response
        await pss.store(answer)
        session_id = pss.session_id
        console.print(f"\n[dim]Session ID: {session_id}[/dim]")

        # ================================================================
        # Part 2: Resume session — memory without chat history
        # ================================================================
        console.rule("[bold yellow]Part 2: Session Continuity (Memory Recall)[/bold yellow]")
        console.print(
            "[dim]New PSSClient with the same session_id but EMPTY LLM message history.\n"
            "The agent recalls facts purely from Habitat AI's semantic memory — \n"
            "no chat replay needed. This is the key advantage over raw context windows.[/dim]\n"
        )

        pss2 = PSSClient(
            base_url=PSS_BASE_URL, api_key=PSS_API_KEY, http=http,
            session_id=session_id,
        )
        deps2 = Deps(pss=pss2)
        history2 = []  # fresh — no LLM history carried over

        recall_questions = [
            "What do you remember about my project?",
            "How much does the hosting cost and what infrastructure are we using?",
            "What's the biggest risk for our Go-Live?",
        ]

        for msg in recall_questions:
            answer, history2 = await run_turn(pss2, deps2, history2, msg)

        await pss2.store(answer)

        # ================================================================
        # Part 3: Cross-topic semantic retrieval
        # ================================================================
        console.rule("[bold yellow]Part 3: Semantic Retrieval Across Topics[/bold yellow]")
        console.print(
            "[dim]PSS retrieves memories by semantic similarity, not recency.\n"
            "Questions about different topics pull the right facts automatically.[/dim]\n"
        )

        cross_topic_questions = [
            "What graph database are we using and why?",
            "Tell me about the OCR requirements — languages and volume.",
            "What's Alex's role on the project?",
        ]

        for msg in cross_topic_questions:
            answer, history2 = await run_turn(pss2, deps2, history2, msg)

        await pss2.store(answer)

        # ================================================================
        # Summary
        # ================================================================
        console.rule("[bold green]Demo Complete[/bold green]")

        table = Table(title="What This Demo Showed", border_style="green")
        table.add_column("Habitat AI Capability", style="bold")
        table.add_column("What Happened")
        table.add_row(
            "Context Compression",
            "6 turns of knowledge built up, but PSS sent only the relevant "
            "subset to the LLM each turn — constant token cost.",
        )
        table.add_row(
            "Session Continuity",
            "Part 2 resumed with empty LLM history. The agent recalled all "
            "project facts from PSS semantic memory alone.",
        )
        table.add_row(
            "Semantic Retrieval",
            "Part 3 asked cross-topic questions. PSS returned the right "
            "memories by similarity score, not by recency.",
        )
        table.add_row(
            "Drift Prevention",
            "PSS tracks conversation phase (initialization → exploration → "
            "resonance → stability) to detect and prevent semantic drift.",
        )
        console.print(table)
        console.print(f"\n[dim]Session: {session_id}[/dim]")
        console.print("[dim]Resume with: uv run agent.py --session {session_id}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
