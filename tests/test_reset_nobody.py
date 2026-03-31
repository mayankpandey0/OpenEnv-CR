"""
tests/test_reset_nobody.py — Verify POST /reset works with no body.

The hackathon automated checker calls POST /reset with no body (null input).
This must return 200 with a valid task, not a 422 validation error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from server.env import app

client = TestClient(app)


def test_reset_with_no_body_returns_200():
    """POST /reset with no body must succeed (checker sends null body)."""
    r = client.post("/reset")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "task_id" in data
    assert "observation" in data
    assert "state" in data
    assert data["state"]["current_step"] == 0
    assert data["state"]["done"] is False


def test_reset_with_empty_json_body_returns_200():
    """POST /reset with empty JSON {} must succeed (uses default seed=42)."""
    r = client.post("/reset", json={})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "task_id" in data
    assert data["seed"] == 42


def test_reset_with_explicit_seed_still_works():
    """POST /reset with explicit seed still works correctly."""
    r = client.post("/reset", json={"seed": 1})
    assert r.status_code == 200
    data = r.json()
    assert data["seed"] == 1


def test_reset_no_body_uses_default_seed_42():
    """POST /reset with no body uses seed=42 (same as explicit seed=42)."""
    r_no_body = client.post("/reset")
    r_seed_42 = client.post("/reset", json={"seed": 42})
    assert r_no_body.status_code == 200
    assert r_seed_42.status_code == 200
    # Both should select the same task
    assert r_no_body.json()["task_id"] == r_seed_42.json()["task_id"]
