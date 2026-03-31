"""
tests/test_env.py — Integration tests for the FastAPI environment server.

Run with:
    pytest tests/test_env.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from server.env import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reset(seed: int = 0) -> dict:
    r = client.post("/reset", json={"seed": seed})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["num_tasks"] >= 1


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------


def test_reset_returns_observation_and_state():
    data = reset(seed=0)
    assert "observation" in data
    assert "state" in data
    obs = data["observation"]
    state = data["state"]
    assert obs["diff_content"]
    assert obs["file_path"]
    assert isinstance(obs["allowed_issue_types"], list)
    assert state["current_step"] == 0
    assert state["done"] is False


def test_reset_is_deterministic():
    d1 = reset(seed=0)
    d2 = reset(seed=0)
    assert d1["task_id"] == d2["task_id"]
    assert d1["observation"]["diff_content"] == d2["observation"]["diff_content"]


def test_reset_different_seeds_give_different_tasks():
    from server.env import TASKS
    if len(TASKS) < 2:
        pytest.skip("Need at least 2 tasks")
    d0 = reset(seed=0)
    d1 = reset(seed=1)
    assert d0["task_id"] != d1["task_id"]


# ---------------------------------------------------------------------------
# /step
# ---------------------------------------------------------------------------


def test_step_without_reset_returns_400():
    # Force uninitialised state by resetting server state manually
    from server.env import GLOBAL_ENV
    GLOBAL_ENV["state"] = None
    GLOBAL_ENV["current_task"] = None
    GLOBAL_ENV["obs"] = None

    r = client.post(
        "/step",
        json={
            "issue_id": "resource_leak",
            "line_number": 5,
            "comment": "Test.",
            "decision": None,
        },
    )
    assert r.status_code == 400


def test_step_valid_hit():
    reset(seed=0)
    from server.env import GLOBAL_ENV
    task = GLOBAL_ENV["current_task"]
    # Find a ground truth entry
    gt = task["ground_truth"]
    line_num = int(next(iter(gt)))
    issue_id = gt[str(line_num)]

    r = client.post(
        "/step",
        json={
            "issue_id": issue_id,
            "line_number": line_num,
            "comment": "Found issue.",
            "decision": None,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["reward"] == pytest.approx(0.35)
    assert data["info"]["valid_hit"] is True
    assert data["done"] is False


def test_step_hallucination():
    reset(seed=0)
    r = client.post(
        "/step",
        json={
            "issue_id": "nonexistent_issue",
            "line_number": 9999,
            "comment": "Bad guess.",
            "decision": None,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["reward"] == pytest.approx(-0.2)
    assert data["info"]["hallucination"] is True


def test_step_decision_ends_episode():
    reset(seed=0)
    r = client.post(
        "/step",
        json={
            "issue_id": "nonexistent_issue",
            "line_number": 1,
            "comment": "Closing.",
            "decision": "REQUEST_CHANGES",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["done"] is True


def test_step_after_done_returns_400():
    reset(seed=0)
    # End the episode
    client.post(
        "/step",
        json={
            "issue_id": "resource_leak",
            "line_number": 1,
            "comment": "Done.",
            "decision": "REQUEST_CHANGES",
        },
    )
    # Try stepping again
    r = client.post(
        "/step",
        json={
            "issue_id": "resource_leak",
            "line_number": 1,
            "comment": "Should fail.",
            "decision": None,
        },
    )
    assert r.status_code == 400


def test_max_steps_ends_episode():
    reset(seed=0)
    from server.env import GLOBAL_ENV
    state = GLOBAL_ENV["state"]
    state.current_step = state.max_steps - 1  # one step remaining

    r = client.post(
        "/step",
        json={
            "issue_id": "resource_leak",
            "line_number": 9999,
            "comment": "Last step.",
            "decision": None,
        },
    )
    assert r.status_code == 200
    assert r.json()["done"] is True


def test_duplicate_issue_gives_zero_reward():
    reset(seed=0)
    from server.env import GLOBAL_ENV
    task = GLOBAL_ENV["current_task"]
    gt = task["ground_truth"]
    line_num = int(next(iter(gt)))
    issue_id = gt[str(line_num)]

    # First hit
    client.post(
        "/step",
        json={"issue_id": issue_id, "line_number": line_num, "comment": "First.", "decision": None},
    )
    # Duplicate
    r = client.post(
        "/step",
        json={"issue_id": issue_id, "line_number": line_num, "comment": "Dup.", "decision": None},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["reward"] == pytest.approx(0.0)
    assert data["info"]["duplicate"] is True


# ---------------------------------------------------------------------------
# /state and /observation
# ---------------------------------------------------------------------------


def test_get_state():
    reset(seed=0)
    r = client.get("/state")
    assert r.status_code == 200
    state = r.json()
    assert "current_step" in state
    assert "done" in state
    assert "found_issue_ids" in state


def test_get_observation():
    reset(seed=0)
    r = client.get("/observation")
    assert r.status_code == 200
    obs = r.json()
    assert "diff_content" in obs
    assert "allowed_issue_types" in obs
    assert "history" in obs


def test_state_before_reset_returns_400():
    from server.env import GLOBAL_ENV
    GLOBAL_ENV["state"] = None
    r = client.get("/state")
    assert r.status_code == 400
