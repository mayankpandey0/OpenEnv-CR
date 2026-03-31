"""
env.py — FastAPI server exposing the OpenEnv-CR environment.

Endpoints:
    POST /reset       — start a new episode (deterministic via seed)
    POST /step        — submit an Action, receive reward + info
    GET  /state       — snapshot of current internal state
    GET  /observation — current observation (diff, allowed types, history)
    GET  /health      — liveness probe for Docker/k8s
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from server.logic import compute_reward
from server.models import Action, Observation, State

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenEnv-CR",
    description=(
        "A fully deterministic, reproducible code-review simulation environment "
        "where an AI agent reviews code diffs and is evaluated via a strict "
        "programmatic grading system."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Task database — loaded once at startup
# ---------------------------------------------------------------------------

TASKS_PATH = Path(__file__).parent / "tasks.json"


def _load_tasks() -> list:
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


TASKS: list = _load_tasks()

# ---------------------------------------------------------------------------
# Global mutable environment state (single-user / single-episode)
# ---------------------------------------------------------------------------

GLOBAL_ENV: Dict[str, Any] = {
    "current_task": None,
    "state": None,
    "obs": None,
}

# ---------------------------------------------------------------------------
# Request / Response models & Classes
# ---------------------------------------------------------------------------

class OpenEnvCREnv:
    def __init__(self):
        pass


class ResetRequest(BaseModel):
    seed: int = 42

    model_config = {"extra": "ignore"}


class ResetResponse(BaseModel):
    observation: dict
    state: dict
    task_id: str
    seed: int


class StepResponse(BaseModel):
    status: str
    proof: str
    details: dict
    reward: float
    info: dict
    done: bool
    observation: dict
    state: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_state(state: State) -> dict:
    """Serialize State, converting found_issue_ids set → sorted list for JSON."""
    d = state.model_dump()
    d["found_issue_ids"] = sorted(d["found_issue_ids"])
    return d


def _require_reset() -> tuple:
    """Return (state, task, obs) or raise 400 if env not initialised."""
    state: Optional[State] = GLOBAL_ENV["state"]
    task: Optional[dict] = GLOBAL_ENV["current_task"]
    obs: Optional[Observation] = GLOBAL_ENV["obs"]
    if state is None or task is None or obs is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )
    return state, task, obs


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index():
    """Redirect root to /docs for Hugging Face UI compatibility."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Meta"])
def health():
    """Liveness probe — always returns 200 OK while the server is up."""
    return {"status": "ok", "num_tasks": len(TASKS)}


@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
def reset(req: Optional[ResetRequest] = None):
    """
    Deterministically select a task using ``seed % num_tasks`` and reset all
    episode state.  Same seed always maps to the same task.
    """
    if not TASKS:
        raise HTTPException(status_code=500, detail="No tasks loaded.")

    seed = req.seed if req is not None else 42
    task_idx = seed % len(TASKS)
    selected_task = TASKS[task_idx]

    state = State(
        current_step=0,
        max_steps=5,
        task_id=selected_task["task_id"],
        found_issue_ids=set(),
        done=False,
    )

    obs = Observation(
        diff_content=selected_task["diff_content"],
        file_path=selected_task["file_path"],
        allowed_issue_types=selected_task["allowed_issue_types"],
        history=[],
    )

    GLOBAL_ENV["current_task"] = selected_task
    GLOBAL_ENV["state"] = state
    GLOBAL_ENV["obs"] = obs

    return ResetResponse(
        observation=obs.model_dump(),
        state=_serialize_state(state),
        task_id=selected_task["task_id"],
        seed=seed,
    )


@app.post("/step", response_model=StepResponse, tags=["Environment"])
def step(payload: Dict[str, Any]):
    """
    Execute one agent action and return the reward, diagnostic info, and the
    updated observation + state.

    Rules enforced here:
    - Raises 400 if the episode is already done.
    - Delegates reward computation to the pure grader in ``logic.py``.
    - Updates ``found_issue_ids``, ``current_step``, and ``done``.
    - Sets ``done=True`` if ``action.decision`` is not None OR ``max_steps`` reached.
    """
    # 1. Intercept test script generic payloads gracefully
    if "issue_id" not in payload and "decision" not in payload:
        return StepResponse(
            status="FAIL", proof="Invalid payload schema detected.", details={},
            reward=0.0, info={}, done=False, observation={}, state={}
        )
    
    try:
        action = Action(**payload)
    except Exception as e:
        return StepResponse(
            status="FAIL", proof=str(e), details={},
            reward=0.0, info={}, done=False, observation={}, state={}
        )

    state, task, obs = _require_reset()

    if state.done:
        raise HTTPException(
            status_code=400,
            detail="Episode already done. Call POST /reset to start a new episode.",
        )

    # ---- Compute reward (pure — no side-effects) ----
    reward, info = compute_reward(
        action=action,
        state=state,
        ground_truth=task["ground_truth"],
        correct_decision=task["correct_decision"],
        allowed_issue_types=task["allowed_issue_types"],
    )

    # ---- Mutate state ----
    if info["valid_hit"]:
        state.found_issue_ids.add(action.issue_id)

    state.current_step += 1

    if action.decision is not None or state.current_step >= state.max_steps:
        state.done = True

    # ---- Append to observation history ----
    obs.history.append(action.model_dump_json())

    return StepResponse(
        status="PASS" if reward > 0 else "FAIL",
        proof=info.get("reason", "Action logged."),
        details={
            "reward": reward,
            "info": info,
            "done": state.done,
            "observation": obs.model_dump(),
            "state": _serialize_state(state),
        },
        reward=reward,
        info=info,
        done=state.done,
        observation=obs.model_dump(),
        state=_serialize_state(state),
    )


@app.get("/state", tags=["Environment"])
def get_state():
    """Return the full current state snapshot (read-only)."""
    state, _, _ = _require_reset()
    return _serialize_state(state)


@app.get("/observation", tags=["Environment"])
def get_observation():
    """Return the current observation (diff, allowed issue types, history)."""
    _, _, obs = _require_reset()
    return obs.model_dump()


def main():
    import uvicorn
    uvicorn.run("server.env:app", host="0.0.0.0", port=7860)
