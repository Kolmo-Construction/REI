"""
Thin HTTP wrapper for PromptFoo evaluation.
Accepts POST /invoke with {query, session_id, store_id, member_number}
Returns {recommendation, action_flag, intent, catalog_results, clarification_message}

Run: uvicorn eval.serve:app --port 8080
"""
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Allow USE_MOCK_LLM to be set via environment before importing greenvest
os.environ.setdefault("USE_MOCK_LLM", os.getenv("USE_MOCK_LLM", "true"))

from greenvest.graph import graph
from greenvest.state import initial_state

app = FastAPI(title="Greenvest Eval Server", version="1.0.0")


class InvokeRequest(BaseModel):
    query: str
    session_id: str = "eval-session"
    store_id: str = "REI-Seattle"
    member_number: Optional[str] = None
    clarification_count: int = 0
    budget_usd: Optional[list] = None


class InvokeResponse(BaseModel):
    recommendation: Optional[str]
    action_flag: str
    intent: Optional[str]
    activity: Optional[str]
    user_environment: Optional[str]
    catalog_results: list
    clarification_message: Optional[str]
    derived_specs: list
    spec_confidence: float


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

        result = await graph.ainvoke(state)

        return InvokeResponse(
            recommendation=result.get("recommendation"),
            action_flag=result.get("action_flag", ""),
            intent=result.get("intent"),
            activity=result.get("activity"),
            user_environment=result.get("user_environment"),
            catalog_results=result.get("catalog_results", []),
            clarification_message=result.get("clarification_message"),
            derived_specs=result.get("derived_specs", []),
            spec_confidence=result.get("spec_confidence", 0.0),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
