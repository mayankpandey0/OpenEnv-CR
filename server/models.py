"""
models.py — Pydantic schemas for OpenEnv-CR.

Defines the Action, Observation, and State data contracts used
throughout the environment, grader, and API.
"""

from enum import Enum
from typing import List, Optional, Set

from pydantic import BaseModel, Field


class Decision(str, Enum):
    """Terminal decision an agent can emit to end an episode."""

    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"


# ---------------------------------------------------------------------------
# Agent-facing schemas
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """A single agent action submitted to /step."""

    issue_id: str = Field(
        description="Must be one of the task's allowed_issue_types."
    )
    line_number: int = Field(
        description="Post-diff line number where the issue was detected."
    )
    comment: str = Field(
        description="Brief human-readable explanation of the issue."
    )
    decision: Optional[Decision] = Field(
        default=None,
        description=(
            "Set to APPROVE or REQUEST_CHANGES to end the episode. "
            "Leave null to continue submitting issues."
        ),
    )


class Observation(BaseModel):
    """Observation returned after /reset and visible during each step."""

    diff_content: str = Field(description="Unified git diff of the file under review.")
    file_path: str = Field(description="Relative path of the file being reviewed.")
    allowed_issue_types: List[str] = Field(
        description="Exhaustive list of issue identifiers the agent may use."
    )
    history: List[str] = Field(
        default_factory=list,
        description="JSON-serialised Action objects from previous steps this episode.",
    )


class State(BaseModel):
    """Full internal state snapshot returned by /state and embedded in /step responses."""

    current_step: int = Field(description="Number of steps taken so far (0-indexed).")
    max_steps: int = Field(description="Hard cap on steps per episode (default 5).")
    task_id: str = Field(description="Identifier of the active task.")
    found_issue_ids: Set[str] = Field(
        default_factory=set,
        description="Set of issue_ids successfully detected in this episode.",
    )
    done: bool = Field(description="True when the episode has ended.")
