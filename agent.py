# /// script
# requires-python = ">=3.10"
# dependencies = ["pydantic-ai", "openai", "httpx", "rich", "anthropic"]
# ///
"""
Example PydanticAI agent with Habitat AI PSS (Persistent Session Store) memory.

Uses the PSS "inline" pattern: each turn calls /run with the previous LLM
response, so PSS stores the Q&A pair and returns relevant context.

Usage:
    uv run agent.py                          # new session
    uv run agent.py --session <session_id>   # resume session
"""

import argparse
import asyncio
import os
from dataclasses import dataclass, field

import httpx
from rich.console import Console
from rich.markdown import Markdown

from pydantic_ai import Agent, RunContext

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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
    MODEL = "openai:gpt-4o-mini"

console = Console()

# ---------------------------------------------------------------------------
# PSS client (thin wrapper)
# ---------------------------------------------------------------------------

@dataclass
class PSSClient:
    base_url: str
    api_key: str
    http: httpx.AsyncClient
    session_id: str | None = None
    _prev_response: str | None = field(default=None, repr=False)

    async def run(self, message: str) -> str:
        """Call PSS /run — returns context string for the system prompt."""
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
        """Call PSS /store — persist the final LLM answer."""
        if not self.session_id:
            return
        resp = await self.http.post(
            f"{self.base_url}/store",
            headers={"X-API-Key": self.api_key},
            json={"session_id": self.session_id, "response": response},
        )
        resp.raise_for_status()

    def set_prev_response(self, response: str) -> None:
        """Buffer the last LLM response so the next /run call sends it."""
        self._prev_response = response


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

@dataclass
class Deps:
    pss: PSSClient
    pss_context: str = ""


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

agent = Agent(
    MODEL,
    deps_type=Deps,
    instructions=(
        "You are a helpful assistant. Answer the user's question using the "
        "conversation context provided below when relevant. "
        "Respond in the same language the user writes in."
    ),
)


@agent.system_prompt
async def inject_pss_context(ctx: RunContext[Deps]) -> str:
    if ctx.deps.pss_context:
        return f"Memory context from previous conversations:\n{ctx.deps.pss_context}"
    return ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main(session_id: str | None = None) -> None:
    async with httpx.AsyncClient(timeout=30) as http:
        pss = PSSClient(
            base_url=PSS_BASE_URL,
            api_key=PSS_API_KEY,
            http=http,
            session_id=session_id,
        )
        deps = Deps(pss=pss)

        if session_id:
            console.print(f"[dim]Resuming session: {session_id}[/dim]")
        else:
            console.print("[dim]Starting new session[/dim]")

        console.print("[dim]Type 'quit' or Ctrl-C to exit.[/dim]\n")

        message_history = []

        try:
            while True:
                user_input = console.input("[bold green]You:[/bold green] ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break

                # 1. Call PSS /run to get context (and store prev response)
                pss_context = await pss.run(user_input)
                deps.pss_context = pss_context

                if pss_context:
                    console.print(f"[dim]{pss_context.splitlines()[0]}[/dim]")

                # 2. Run the agent
                result = await agent.run(
                    user_input,
                    deps=deps,
                    message_history=message_history,
                )
                answer = result.output

                # 3. Display the response
                console.print()
                console.print(Markdown(answer))
                console.print()

                # 4. Buffer the response so next /run stores it
                pss.set_prev_response(answer)

                # 5. Keep message history for multi-turn LLM context
                message_history = result.all_messages()

        except (KeyboardInterrupt, EOFError):
            pass

        # Store the final response before exiting
        if pss._prev_response:
            await pss.store(pss._prev_response)

        console.print(f"\n[dim]Session: {pss.session_id}[/dim]")
        console.print("[dim]Goodbye![/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PydanticAI agent with PSS memory")
    parser.add_argument("--session", type=str, default=None, help="Resume a PSS session by ID")
    args = parser.parse_args()
    asyncio.run(main(session_id=args.session))
