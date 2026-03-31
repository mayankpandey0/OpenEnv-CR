"""
tests/test_judge_stress.py — Judge's Stress Test
=================================================
Validates that the environment correctly handles three exploit patterns:

  Scenario 1 — The Spammer:
      Submit a valid hit once, then repeat the identical action 4 more times.
      Duplicate-detection must clamp total reward to a single hit (no farming).

  Scenario 2 — The Guessing Bot:
      Submit 5 actions with random/nonsense line numbers and valid issue_ids.
      The -0.2 hallucination penalty must accumulate to a large negative total.

  Scenario 3 — The Speedrunner:
      Submit decision="APPROVE" on Step 1 (with a bogus issue payload).
      Environment must set done=True immediately and the correct terminal
      decision must be evaluated — no further steps allowed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from server.env import app, GLOBAL_ENV, TASKS

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def do_reset(seed: int = 0) -> dict:
    r = client.post("/reset", json={"seed": seed})
    assert r.status_code == 200, f"Reset failed: {r.text}"
    return r.json()


def do_step(payload: dict) -> dict:
    r = client.post("/step", json=payload)
    return r


# ---------------------------------------------------------------------------
# Scenario 1: The Spammer
# ---------------------------------------------------------------------------

class TestSpammer:
    """
    The Spammer submits a valid hit on Step 1, then repeats the EXACT same
    JSON action for Steps 2-5.

    Expected behaviour:
    - Step 1 → valid hit → reward = 0.7 / len(ground_truth)  (NOT 0.5)
    - Steps 2-5 → duplicate → reward = 0.0 each
    - Total reward = single hit value (not multiplied by 5)
    """

    def test_spammer_total_reward_is_single_hit(self):
        data = do_reset(seed=0)
        task = GLOBAL_ENV["current_task"]
        gt = task["ground_truth"]

        # Build the valid hit action
        line_str, issue_id = next(iter(gt.items()))
        valid_action = {
            "issue_id": issue_id,
            "line_number": int(line_str),
            "comment": "Found an issue.",
            "decision": None,
        }

        total_reward = 0.0
        rewards_per_step = []
        info_per_step = []

        # Submit 5 times (same action)
        for step_idx in range(1, 6):
            r = do_step(valid_action)
            if r.status_code == 400:
                # Episode already done — that's acceptable
                print(f"  Step {step_idx}: episode done (400), stopping.")
                break
            assert r.status_code == 200, f"Unexpected error on step {step_idx}: {r.text}"
            sd = r.json()
            total_reward += sd["reward"]
            rewards_per_step.append(sd["reward"])
            info_per_step.append(sd["info"])
            print(
                f"  Step {step_idx}: reward={sd['reward']:+.4f} "
                f"  valid_hit={sd['info']['valid_hit']}  "
                f"  duplicate={sd['info']['duplicate']}  "
                f"  reason={sd['info']['reason']}"
            )

        expected_single_hit = 0.7 / max(1, len(gt))
        print(f"\n  Spammer total reward = {total_reward:+.4f}")
        print(f"  Expected single-hit  = {expected_single_hit:+.4f}")
        print(f"  Rewards per step     = {rewards_per_step}")

        # Step 1 must be a valid hit
        assert info_per_step[0]["valid_hit"] is True, (
            "Step 1 should be a valid hit"
        )

        # Steps 2+ must all be duplicates (zero reward)
        for i, info in enumerate(info_per_step[1:], 2):
            assert info["duplicate"] is True, (
                f"Step {i} should be a duplicate, got: {info}"
            )
            assert rewards_per_step[i - 1] == pytest.approx(0.0), (
                f"Step {i} reward should be 0.0 (duplicate), got {rewards_per_step[i-1]}"
            )

        # Total reward must equal exactly one valid hit, not 5x
        assert total_reward == pytest.approx(expected_single_hit, abs=1e-6), (
            f"EXPLOIT DETECTED: Spammer farmed {total_reward:.4f} "
            f"instead of {expected_single_hit:.4f}"
        )

    def test_spammer_cannot_exceed_max_steps(self):
        """Even without duplicate protection, 5-step cap limits spam."""
        do_reset(seed=0)
        task = GLOBAL_ENV["current_task"]
        gt = task["ground_truth"]
        line_str, issue_id = next(iter(gt.items()))
        valid_action = {
            "issue_id": issue_id,
            "line_number": int(line_str),
            "comment": "Spam attempt.",
            "decision": None,
        }

        step_count = 0
        for _ in range(10):  # attempt more than max_steps
            r = do_step(valid_action)
            if r.status_code == 400:
                break
            step_count += 1

        print(f"\n  Spammer stopped after {step_count} accepted steps (max=5)")
        assert step_count <= 5, f"EXPLOIT: accepted {step_count} steps beyond max_steps=5"


# ---------------------------------------------------------------------------
# Scenario 2: The Guessing Bot
# ---------------------------------------------------------------------------

class TestGuessingBot:
    """
    The Guessing Bot submits 5 actions with random/nonsense line numbers
    (1, 99, 500, 1000, 9999) and valid issue_ids.

    Expected behaviour:
    - Each wrong line → -0.2 hallucination penalty
    - 5 wrong guesses → total ≈ -1.0
    - Total reward must be significantly negative
    """

    RANDOM_LINES = [1, 99, 500, 1000, 9999]

    def test_guessing_bot_accumulates_penalty(self):
        data = do_reset(seed=0)
        task = GLOBAL_ENV["current_task"]
        allowed = task["allowed_issue_types"]
        gt = task["ground_truth"]

        total_reward = 0.0
        print()
        for step_idx, line_num in enumerate(self.RANDOM_LINES, 1):
            # Confirm none of the random lines are actually in ground truth
            is_in_gt = str(line_num) in gt
            action = {
                "issue_id": allowed[0],   # valid issue type
                "line_number": line_num,  # random / wrong line
                "comment": f"Random guess at line {line_num}.",
                "decision": None,
            }
            r = do_step(action)
            if r.status_code == 400:
                print(f"  Step {step_idx}: episode done (400)")
                break
            assert r.status_code == 200
            sd = r.json()
            total_reward += sd["reward"]
            print(
                f"  Step {step_idx}: line={line_num} in_gt={is_in_gt}  "
                f"reward={sd['reward']:+.4f}  "
                f"hallucination={sd['info']['hallucination']}  "
                f"reason={sd['info']['reason']}"
            )

        print(f"\n  Guessing bot total reward = {total_reward:+.4f}")
        assert total_reward < -0.5, (
            f"EXPLOIT: Guessing bot accumulated only {total_reward:.4f}; "
            f"expected significantly negative (< -0.5)"
        )

    def test_guessing_bot_each_step_penalised(self):
        """Every wrong-line guess must individually yield -0.2."""
        do_reset(seed=0)
        task = GLOBAL_ENV["current_task"]
        allowed = task["allowed_issue_types"]
        gt = task["ground_truth"]

        # Pick lines guaranteed to be absent from ground truth
        bogus_lines = [l for l in [500, 1000, 9999] if str(l) not in gt]

        print()
        for line_num in bogus_lines[:3]:
            action = {
                "issue_id": allowed[0],
                "line_number": line_num,
                "comment": f"Bogus line {line_num}.",
                "decision": None,
            }
            r = do_step(action)
            if r.status_code == 400:
                break
            sd = r.json()
            print(
                f"  Bogus line {line_num}: reward={sd['reward']:+.4f}  "
                f"hallucination={sd['info']['hallucination']}"
            )
            assert sd["reward"] == pytest.approx(-0.2), (
                f"Expected -0.2 for bogus line {line_num}, got {sd['reward']}"
            )
            assert sd["info"]["hallucination"] is True


# ---------------------------------------------------------------------------
# Scenario 3: The Speedrunner
# ---------------------------------------------------------------------------

class TestSpeedrunner:
    """
    The Speedrunner submits a terminal decision on Step 1 without doing any
    real issue identification.

    Expected behaviour:
    - done=True is set immediately after Step 1
    - The environment evaluates the decision (correct or not) and returns reward
    - A second /step call must return HTTP 400 (episode done)
    """

    def test_speedrunner_done_true_after_step1_with_decision(self):
        data = do_reset(seed=0)
        task = GLOBAL_ENV["current_task"]
        allowed = task["allowed_issue_types"]

        # Speedrunner skips issue hunting; jumps straight to APPROVE
        action = {
            "issue_id": allowed[0],  # required field — bogus but structurally valid
            "line_number": 9999,     # wrong line
            "comment": "Speedrunning to decision.",
            "decision": "APPROVE",   # no issue found — just fire terminal decision
        }

        r = do_step(action)
        assert r.status_code == 200, f"Step 1 failed: {r.text}"
        sd = r.json()
        print(f"\n  Speedrunner Step 1: reward={sd['reward']:+.4f}  done={sd['done']}")
        print(f"  reason: {sd['info']['reason']}")

        # Environment MUST set done=True immediately
        assert sd["done"] is True, (
            f"EXPLOIT: Speedrunner's done=False after submitting decision; "
            f"agent can keep stepping!"
        )

    def test_speedrunner_blocked_after_done(self):
        """Once done=True, any further /step must return 400."""
        do_reset(seed=0)
        task = GLOBAL_ENV["current_task"]
        allowed = task["allowed_issue_types"]

        # End episode immediately
        r1 = do_step({
            "issue_id": allowed[0],
            "line_number": 9999,
            "comment": "Ending now.",
            "decision": "APPROVE",
        })
        assert r1.json()["done"] is True

        # Attempt second step — must be rejected
        r2 = do_step({
            "issue_id": allowed[0],
            "line_number": 9999,
            "comment": "Should be rejected.",
            "decision": None,
        })
        print(f"\n  Second step after done: status={r2.status_code}")
        assert r2.status_code == 400, (
            f"EXPLOIT: Speedrunner can submit steps after done=True! "
            f"Got status {r2.status_code}"
        )

    def test_speedrunner_correct_decision_evaluates_reward(self):
        """Correct APPROVE decision gives +0.3, wrong APPROVE gives 0."""
        # task easy_syntax_1 (seed=0) correct_decision = REQUEST_CHANGES
        data = do_reset(seed=0)
        task = GLOBAL_ENV["current_task"]
        allowed = task["allowed_issue_types"]
        correct_decision = task["correct_decision"]

        print(f"\n  Task correct_decision = '{correct_decision}'")

        # Speedrunner submits the WRONG terminal decision (APPROVE vs REQUEST_CHANGES)
        r = do_step({
            "issue_id": allowed[0],
            "line_number": 9999,
            "comment": "Speedrun with wrong decision.",
            "decision": "APPROVE",  # wrong — task needs REQUEST_CHANGES
        })
        sd = r.json()
        print(f"  Speedrunner (wrong decision): reward={sd['reward']:+.4f}  done={sd['done']}")
        print(f"  correct_decision flag: {sd['info']['correct_decision']}")

        # Wrong decision + hallucination = -0.2 only (no +0.3 bonus)
        assert sd["info"]["correct_decision"] is False
        assert sd["done"] is True

    def test_speedrunner_with_correct_decision_gets_bonus(self):
        """Speedrunner with correct terminal decision still gets +0.3 (minus hallucination)."""
        data = do_reset(seed=0)
        task = GLOBAL_ENV["current_task"]
        allowed = task["allowed_issue_types"]
        correct_decision = task["correct_decision"]  # "REQUEST_CHANGES"

        r = do_step({
            "issue_id": allowed[0],
            "line_number": 9999,
            "comment": "Speedrun with correct decision.",
            "decision": correct_decision,  # correct
        })
        sd = r.json()
        print(f"\n  Speedrunner (correct decision): reward={sd['reward']:+.4f}  done={sd['done']}")
        print(f"  reason: {sd['info']['reason']}")

        # Expected: -0.2 (hallucination) + 0.3 (correct decision) = 0.1
        assert sd["reward"] == pytest.approx(0.1, abs=1e-6), (
            f"Expected 0.1 but got {sd['reward']}"
        )
        assert sd["info"]["correct_decision"] is True
        assert sd["done"] is True


# ---------------------------------------------------------------------------
# Scenario 4: Combined / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Additional boundary conditions exposed by the judge scenarios."""

    def test_reward_formula_uses_scaled_hit_not_flat_05(self):
        """
        The reward formula in logic.py uses 0.7/len(gt), NOT a flat 0.5.
        This test documents the actual formula in play.
        """
        do_reset(seed=0)
        task = GLOBAL_ENV["current_task"]
        gt = task["ground_truth"]
        line_str, issue_id = next(iter(gt.items()))

        r = do_step({
            "issue_id": issue_id,
            "line_number": int(line_str),
            "comment": "Checking reward formula.",
            "decision": None,
        })
        sd = r.json()
        expected = 0.7 / max(1, len(gt))
        print(f"\n  Hit reward: got={sd['reward']:.4f}  expected=0.7/len(gt)={expected:.4f}")
        assert sd["reward"] == pytest.approx(expected, abs=1e-6)

    def test_out_of_vocabulary_issue_id_always_penalised(self):
        """An issue_id not in allowed_issue_types triggers -0.2 regardless of line."""
        do_reset(seed=0)
        r = do_step({
            "issue_id": "sql_injection_FAKE",   # OOV
            "line_number": 2,                    # could be valid line in task
            "comment": "OOV test.",
            "decision": None,
        })
        sd = r.json()
        print(f"\n  OOV issue_id: reward={sd['reward']:+.4f}  hallucination={sd['info']['hallucination']}")
        assert sd["reward"] == pytest.approx(-0.2)
        assert sd["info"]["hallucination"] is True
