"""
Interactive agent runner — conversational REPL.

Usage:
    uv run python scripts/run_agent.py                  # start fresh session
    uv run python scripts/run_agent.py "first message"  # start with an initial query
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from greenvest.graph import graph
from greenvest.state import initial_state, GreenvestState


def _print_result(result: dict) -> None:
    print(f"\nINTENT:    {result.get('intent')}")
    print(f"ACTIVITY:  {result.get('activity')}")
    print(f"ENV:       {result.get('user_environment')}")
    print(f"FLAG:      {result.get('action_flag')}")

    if result.get("derived_specs"):
        print(f"SPECS:     {result['derived_specs']}")

    if result.get("catalog_results"):
        print(f"\nPRODUCTS FOUND ({len(result['catalog_results'])}):")
        for p in result["catalog_results"]:
            print(f"  - {p['name']} | ${p['price_usd']} | {p.get('fill_type','N/A')} | {p.get('temp_rating_f','N/A')}°F")

    if result.get("expert_context"):
        print(f"\nEXPERT CHUNKS USED ({len(result['expert_context'])}):")
        for c in result["expert_context"]:
            print(f"  [{c.get('section','')}] score={c.get('score',0):.3f}")

    if result.get("clarification_message"):
        print(f"\nAGENT: {result['clarification_message']}")

    if result.get("recommendation"):
        print(f"\n{'-'*60}")
        print("AGENT:")
        print(f"{'-'*60}")
        print(result["recommendation"])

    print()


async def chat_loop(first_query: str | None = None) -> None:
    """Run a multi-turn conversation, carrying state between turns."""
    print("\nGreenvest Agent  (type 'quit' or Ctrl-C to exit)\n")

    state: GreenvestState | None = None

    while True:
        # Get user input
        if first_query:
            query = first_query
            first_query = None
        else:
            try:
                query = input("YOU: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not query:
            continue

        print(f"\n{'='*60}")
        print(f"CUSTOMER: {query}")
        print(f"{'='*60}")

        if state is None:
            # Fresh session
            state = initial_state(query=query)
        else:
            # Continue session — update query, preserve accumulated context
            state = {
                **state,
                "query": query,
                # Reset per-turn output fields
                "clarification_message": None,
                "recommendation": None,
                "derived_specs": [],
                "expert_context": [],
                "catalog_results": [],
                "inventory_snapshot": [],
                "action_flag": "REQUIRES_CLARIFICATION",
            }

        state = await graph.ainvoke(state)
        _print_result(state)

        # If the turn produced a full recommendation, offer to start fresh
        if state.get("recommendation"):
            print("(Start a new question or type 'quit' to exit)\n")
            state = None  # Reset for next independent question


def main() -> None:
    first_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    asyncio.run(chat_loop(first_query))


if __name__ == "__main__":
    main()
