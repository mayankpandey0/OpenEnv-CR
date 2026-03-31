"""
tests/test_logic.py — Unit tests for the pure reward grader.

Run with:
    pytest tests/test_logic.py -v
"""

import sys
from pathlib import Path

# Ensure the project root is on the import path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from server.models import Action, Decision, State
from server.logic import compute_reward

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GROUND_TRUTH = {"5": "resource_leak", "9": "missing_colon"}
CORRECT_DECISION = "REQUEST_CHANGES"
ALLOWED = ["resource_leak", "missing_colon", "unhandled_exception"]


def _base_state(**overrides) -> State:
    defaults = dict(
        current_step=0,
        max_steps=5,
        task_id="test_task",
        found_issue_ids=set(),
        done=False,
    )
    defaults.update(overrides)
    return State(**defaults)


def _reward(action: Action, state: State = None) -> tuple:
    if state is None:
        state = _base_state()
    return compute_reward(
        action=action,
        state=state,
        ground_truth=GROUND_TRUTH,
        correct_decision=CORRECT_DECISION,
        allowed_issue_types=ALLOWED,
    )


# ---------------------------------------------------------------------------
# Valid hit
# ---------------------------------------------------------------------------


def test_valid_hit_gives_positive_reward():
    action = Action(issue_id="resource_leak", line_number=5, comment="File not closed.")
    reward, info = _reward(action)
    assert reward == pytest.approx(0.35)
    assert info["valid_hit"] is True
    assert info["hallucination"] is False
    assert info["duplicate"] is False


def test_valid_hit_second_issue():
    action = Action(issue_id="missing_colon", line_number=9, comment="Missing colon.")
    reward, info = _reward(action)
    assert reward == pytest.approx(0.35)
    assert info["valid_hit"] is True


# ---------------------------------------------------------------------------
# Hallucination
# ---------------------------------------------------------------------------


def test_wrong_line_is_hallucination():
    action = Action(issue_id="resource_leak", line_number=99, comment="Wrong line.")
    reward, info = _reward(action)
    assert reward == pytest.approx(-0.2)
    assert info["hallucination"] is True
    assert info["valid_hit"] is False


def test_correct_line_wrong_issue_id_is_hallucination():
    action = Action(issue_id="missing_colon", line_number=5, comment="Wrong label.")
    reward, info = _reward(action)
    assert reward == pytest.approx(-0.2)
    assert info["hallucination"] is True


def test_out_of_vocabulary_issue_id_is_hallucination():
    action = Action(issue_id="sql_injection", line_number=5, comment="OOV type.")
    reward, info = _reward(action)
    assert reward == pytest.approx(-0.2)
    assert info["hallucination"] is True


# ---------------------------------------------------------------------------
# Duplicate
# ---------------------------------------------------------------------------


def test_duplicate_gives_zero_reward():
    state = _base_state(found_issue_ids={"resource_leak"})
    action = Action(issue_id="resource_leak", line_number=5, comment="Already found.")
    reward, info = _reward(action, state=state)
    assert reward == pytest.approx(0.0)
    assert info["duplicate"] is True
    assert info["valid_hit"] is False
    assert info["hallucination"] is False


# ---------------------------------------------------------------------------
# Terminal decision
# ---------------------------------------------------------------------------


def test_correct_decision_adds_0_3():
    action = Action(
        issue_id="resource_leak",
        line_number=5,
        comment="Closing.",
        decision=Decision.REQUEST_CHANGES,
    )
    reward, info = _reward(action)
    # valid hit (0.35) + correct decision (0.3) = 0.65
    assert reward == pytest.approx(0.65)
    assert info["correct_decision"] is True


def test_incorrect_decision_no_bonus():
    action = Action(
        issue_id="resource_leak",
        line_number=5,
        comment="Closing.",
        decision=Decision.APPROVE,  # wrong — should be REQUEST_CHANGES
    )
    reward, info = _reward(action)
    assert reward == pytest.approx(0.35)  # only valid hit
    assert info["correct_decision"] is False


def test_hallucination_with_correct_decision():
    """Agent gets line wrong but issues the right terminal decision."""
    action = Action(
        issue_id="resource_leak",
        line_number=99,
        comment="Wrong line.",
        decision=Decision.REQUEST_CHANGES,
    )
    reward, info = _reward(action)
    # -0.2 (hall) + 0.3 (correct decision) = 0.1
    assert reward == pytest.approx(0.1)
    assert info["hallucination"] is True
    assert info["correct_decision"] is True


# ---------------------------------------------------------------------------
# Info dict completeness
# ---------------------------------------------------------------------------


def test_info_dict_has_all_keys():
    action = Action(issue_id="resource_leak", line_number=5, comment="Test.")
    _, info = _reward(action)
    for key in ("valid_hit", "hallucination", "duplicate", "correct_decision", "reason"):
        assert key in info, f"Missing key: {key}"


def test_reason_is_non_empty():
    action = Action(issue_id="resource_leak", line_number=5, comment="Test.")
    _, info = _reward(action)
    assert isinstance(info["reason"], str)
    assert len(info["reason"]) > 0
