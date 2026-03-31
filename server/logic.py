"""
logic.py — Pure, stateless reward grader for OpenEnv-CR.

Reward formula (as specified):

    R = (0.5 × ValidHit) − (0.2 × Hallucination) + (0.3 × CorrectDecision)

This module has NO side effects — it only reads from its arguments.
All state mutation is owned by env.py.
"""

from typing import Any, Dict, Tuple

from server.models import Action, State


def compute_reward(
    action: Action,
    state: State,
    ground_truth: Dict[str, str],
    correct_decision: str,
    allowed_issue_types: list,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the per-step reward and diagnostic info for one agent action.

    Args:
        action:             The agent's submitted action.
        state:              Current environment state (read-only here).
        ground_truth:       Mapping of str(line_number) → expected issue_id.
        correct_decision:   The task's ground-truth terminal decision string.
        allowed_issue_types: Valid issue identifiers for this task.

    Returns:
        A (reward, info) tuple where info contains:
            valid_hit       bool  — True when line + issue_id both correct and new.
            hallucination   bool  — True when line or issue_id is wrong.
            duplicate       bool  — True when issue_id was already found.
            correct_decision bool — True when terminal decision matches ground truth.
            reason          str   — Human-readable explanation of the outcome.
    """
    reward = 0.0
    info: Dict[str, Any] = {
        "valid_hit": False,
        "hallucination": False,
        "duplicate": False,
        "correct_decision": False,
        "reason": "",
    }

    reasons: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. Validate issue_id against task's allowed_issue_types FIRST.      #
    #    An out-of-vocabulary ID is a hallucination regardless of line.   #
    # ------------------------------------------------------------------ #
    if action.issue_id not in allowed_issue_types:
        reward -= 0.2
        info["hallucination"] = True
        reasons.append(
            f"issue_id '{action.issue_id}' is not in allowed_issue_types "
            f"{allowed_issue_types}."
        )
        # Still evaluate the decision below — agent can still get partial reward.

    else:
        # --------------------------------------------------------------- #
        # 2. Issue evaluation: line_number + issue_id match check.        #
        # --------------------------------------------------------------- #
        line_str = str(action.line_number)

        if line_str in ground_truth:
            expected_issue = ground_truth[line_str]
            if action.issue_id == expected_issue:
                if action.issue_id not in state.found_issue_ids:
                    # ✓ Valid, new hit
                    hit_reward = 0.7 / max(1, len(ground_truth))
                    reward += hit_reward
                    info["valid_hit"] = True
                    reasons.append(
                        f"Valid hit: line {action.line_number} -> '{action.issue_id}'."
                    )
                else:
                    # Duplicate — zero reward, no penalty
                    info["duplicate"] = True
                    reasons.append(
                        f"Duplicate: '{action.issue_id}' was already found — "
                        "zero reward."
                    )
            else:
                # Correct line, wrong issue label
                reward -= 0.2
                info["hallucination"] = True
                reasons.append(
                    f"Wrong issue_id on line {action.line_number}: "
                    f"expected '{expected_issue}', got '{action.issue_id}'."
                )
        else:
            # Line number not in ground truth
            reward -= 0.2
            info["hallucination"] = True
            reasons.append(
                f"Line {action.line_number} has no issue in ground_truth."
            )

    # ------------------------------------------------------------------ #
    # 3. Terminal decision evaluation (only when agent sets decision).    #
    # ------------------------------------------------------------------ #
    if action.decision is not None:
        if action.decision.value == correct_decision:
            reward += 0.3
            info["correct_decision"] = True
            reasons.append(
                f"Correct terminal decision: '{action.decision.value}'."
            )
        else:
            reasons.append(
                f"Incorrect decision: expected '{correct_decision}', "
                f"got '{action.decision.value}'."
            )

    info["reason"] = " ".join(reasons) if reasons else "No evaluable action."

    return reward, info
