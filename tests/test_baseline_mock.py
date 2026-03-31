"""
tests/test_baseline_mock.py
===========================
End-to-end mock test for baseline_agent.py logic.

Simulates what a *perfect* GPT-4o would output for each task
(answers sourced directly from tasks.json ground truth) and verifies
that the environment rewards the correct score without requiring a
live LLM API key.

Run:
    pytest tests/test_baseline_mock.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from fastapi.testclient import TestClient
from server.env import app, TASKS

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper: build perfect answers from a task's ground_truth
# ---------------------------------------------------------------------------

def max_achievable_score(task: dict) -> float:
    """
    Compute the theoretical maximum score for a task.

    The environment tracks found_issue_ids as a *set of issue_id strings*,
    NOT (line, issue_id) pairs.  Therefore if two ground-truth entries share
    the same issue_id only the FIRST submission scores a valid hit; subsequent
    ones are duplicates (0 reward, no penalty).

    Max score = (unique_issue_ids × hit_reward_each) + 0.3 decision bonus
    where hit_reward_each = 0.7 / len(ground_truth)
    """
    gt = task["ground_truth"]
    hit_value = 0.7 / max(1, len(gt))
    unique_ids = set(gt.values())
    max_hit_total = hit_value * len(unique_ids)
    return min(1.0, max_hit_total + 0.3)


def perfect_actions_for_task(task: dict) -> list[dict]:
    """
    Generate the optimal sequence of actions for a task.

    Strategy:
    - Submit each *unique* issue_id once at its first matching line number.
    - On the final submission attach the correct terminal decision.

    NOTE: The environment's duplicate-detection is keyed on issue_id alone
    (a set), so submitting the same issue_id twice wastes a step.  We skip
    duplicate issue_ids after the first occurrence.
    """
    gt: dict = task["ground_truth"]
    items = sorted(gt.items(), key=lambda kv: int(kv[0]))  # sort by line number

    seen_ids: set[str] = set()
    unique_items = []
    for line_str, issue_id in items:
        if issue_id not in seen_ids:
            seen_ids.add(issue_id)
            unique_items.append((line_str, issue_id))

    actions = []
    for i, (line_str, issue_id) in enumerate(unique_items):
        is_last = i == len(unique_items) - 1
        actions.append({
            "issue_id": issue_id,
            "line_number": int(line_str),
            "comment": f"Found {issue_id} at line {line_str}.",
            "decision": task["correct_decision"] if is_last else None,
        })
    return actions


# ---------------------------------------------------------------------------
# Per-task tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(range(len(TASKS))))
def test_perfect_agent_scores_high(seed: int):
    """
    An optimal agent should achieve the maximum possible score for each task.

    The environment tracks found_issue_ids as a set of issue_id strings, so
    tasks where two ground-truth entries share the same issue_id cap out below
    1.0.  We assert against max_achievable_score() rather than a fixed 0.9.
    """
    r = client.post("/reset", json={"seed": seed})
    assert r.status_code == 200, r.text
    data = r.json()
    task_id = data["task_id"]

    # Find the actual task object
    task = next(t for t in TASKS if t["task_id"] == task_id)
    actions = perfect_actions_for_task(task)
    expected_max = max_achievable_score(task)

    episode_reward = 0.0
    for i, action in enumerate(actions, 1):
        sr = client.post("/step", json=action)
        assert sr.status_code == 200, f"Step {i} failed: {sr.text}"
        sd = sr.json()
        episode_reward += sd["reward"]

        print(
            f"  [{task_id}] Step {i}: "
            f"reward={sd['reward']:+.3f}  reason={sd['info']['reason']}"
        )

        if sd["done"]:
            break

    # Clamp as the spec requires
    score = max(0.0, min(1.0, episode_reward))
    print(f"  [{task_id}] FINAL SCORE = {score:.3f}  (max possible = {expected_max:.3f})")

    # Score should exactly equal the theoretical maximum achievable
    assert score == pytest.approx(expected_max, abs=1e-6), (
        f"Expected score={expected_max:.3f} but got {score:.3f}"
    )


def test_hallucinating_agent_scores_low():
    """
    An agent that always guesses wrong line numbers should score <= 0.
    Verifies the -0.2 penalty accumulates correctly.
    """
    client.post("/reset", json={"seed": 0})
    task = TASKS[0]
    allowed = task["allowed_issue_types"]

    episode_reward = 0.0
    for step in range(3):
        is_last = step == 2
        action = {
            "issue_id": allowed[0],        # valid issue type
            "line_number": 9999,           # always wrong line → hallucination
            "comment": "Wrong guess",
            "decision": "REQUEST_CHANGES" if is_last else None,
        }
        sr = client.post("/step", json=action)
        sd = sr.json()
        episode_reward += sd["reward"]
        if sd["done"]:
            break

    # -0.2 × 2 wrong lines + 0.3 correct decision = -0.1 → clamped to 0.0
    score = max(0.0, min(1.0, episode_reward))
    print(f"  Hallucinating agent score = {score:.3f}")
    assert score == pytest.approx(0.0)


def test_overall_average_across_all_tasks():
    """
    Aggregate score across all tasks with an optimal agent should equal the
    average of each task's max_achievable_score().
    """
    scores = []
    max_scores = []
    for seed in range(len(TASKS)):
        r = client.post("/reset", json={"seed": seed})
        data = r.json()
        task = next(t for t in TASKS if t["task_id"] == data["task_id"])
        actions = perfect_actions_for_task(task)
        max_scores.append(max_achievable_score(task))

        ep_reward = 0.0
        for action in actions:
            sr = client.post("/step", json=action)
            sd = sr.json()
            ep_reward += sd["reward"]
            if sd["done"]:
                break

        scores.append(max(0.0, min(1.0, ep_reward)))

    avg = sum(scores) / len(scores)
    expected_avg = sum(max_scores) / len(max_scores)
    print(f"\n  All-task average (optimal agent)  = {avg:.3f}")
    print(f"  Theoretical max average           = {expected_avg:.3f}")
    assert avg == pytest.approx(expected_avg, abs=1e-6), (
        f"Expected average={expected_avg:.3f}, got {avg:.3f}"
    )
