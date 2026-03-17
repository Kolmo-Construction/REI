"""
Thin HTTP wrapper for PromptFoo evaluation.
Accepts POST /invoke with {query, session_id, store_id, member_number}
Returns {recommendation, action_flag, intent, catalog_results, clarification_message}

Run: uvicorn eval.serve:app --port 8080
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Allow USE_MOCK_LLM to be set via environment before importing greenvest
os.environ.setdefault("USE_MOCK_LLM", os.getenv("USE_MOCK_LLM", "false"))

from greenvest.logging_config import configure_logging
configure_logging()

from greenvest.graph import graph
from greenvest.state import initial_state
from eval.langfuse_client import langfuse_client

app = FastAPI(title="Greenvest Eval Server", version="1.0.0")


class InvokeRequest(BaseModel):
    query: str
    session_id: str = "eval-session"
    store_id: str = "REI-Seattle"
    member_number: Optional[str] = None
    clarification_count: int = 0
    budget_usd: Optional[list] = None
    pre_seed_state: dict = {}


class InvokeResponse(BaseModel):
    recommendation: Optional[str]
    action_flag: str
    intent: Optional[str]
    activity: Optional[str]
    user_environment: Optional[str]
    catalog_results: list
    clarification_message: Optional[str]
    derived_specs: dict
    spec_confidence: float
    messages: list


_FEEDBACK_PATH = Path(__file__).parent.parent / "experiments" / "user_feedback.jsonl"


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    response: str
    action_flag: str
    intent: Optional[str] = None
    activity: Optional[str] = None
    store_id: str = "REI-Seattle"
    vote: str  # "up" or "down"


@app.post("/feedback", status_code=204)
async def feedback(req: FeedbackRequest):
    """Append a thumbs-up/down vote to experiments/user_feedback.jsonl."""
    if req.vote not in ("up", "down"):
        raise HTTPException(status_code=400, detail="vote must be 'up' or 'down'")
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": req.session_id,
        "store_id": req.store_id,
        "query": req.query,
        "response": req.response,
        "action_flag": req.action_flag,
        "intent": req.intent,
        "activity": req.activity,
        "vote": req.vote,
    }
    _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health")
async def health():
    """Health check endpoint for load balancers and CI."""
    return {"status": "ok", "use_mock_llm": os.getenv("USE_MOCK_LLM", "true")}


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest):
    """Run the Greenvest agent graph and return the result."""
    try:
        state = initial_state(
            query=req.query,
            session_id=req.session_id,
            store_id=req.store_id,
            member_number=req.member_number,
        )
        state["clarification_count"] = req.clarification_count
        if req.budget_usd is not None:
            state["budget_usd"] = tuple(req.budget_usd)
        # Re-inject previously collected state for multi-turn conversations
        for k, v in req.pre_seed_state.items():
            state[k] = v

        lf = langfuse_client()
        if lf:
            with lf.start_as_current_observation(
                name="greenvest_invoke",
                as_type="span",
                input=req.model_dump(),
            ) as span:
                result = await graph.ainvoke(state)
                span.update(output={
                    "action_flag": result.get("action_flag"),
                    "intent": result.get("intent"),
                    "recommendation": result.get("recommendation"),
                })
        else:
            result = await graph.ainvoke(state)

        # Serialise messages: LangChain BaseMessage objects are not JSON-serialisable
        raw_messages = result.get("messages", [])
        serialised_messages = [
            {"type": getattr(m, "type", "unknown"), "content": getattr(m, "content", str(m))}
            for m in raw_messages
        ]

        return InvokeResponse(
            recommendation=result.get("recommendation"),
            action_flag=result.get("action_flag", ""),
            intent=result.get("intent"),
            activity=result.get("activity"),
            user_environment=result.get("user_environment"),
            catalog_results=result.get("catalog_results", []),
            clarification_message=result.get("clarification_message"),
            derived_specs=result.get("derived_specs", {}),
            spec_confidence=result.get("spec_confidence", 0.0),
            messages=serialised_messages,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Must be registered last so API routes always take priority over static files
_WEB_DIR = Path(__file__).parent.parent / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="static")
