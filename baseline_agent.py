"""
baseline_agent.py — GPT-4o Baseline Agent for OpenEnv-CR
==========================================================

Goal: Verify that a standard LLM (GPT-4o) can understand code diffs and
successfully navigate the reward system to achieve a high score.

System prompt mirrors the task specification exactly:
  - Analysis Phase:  identify exact line number + issue type from allowed_issue_types
  - Action Phase:    submit findings one by one (issue_id + line_number)
  - Decision Phase:  submit APPROVE or REQUEST_CHANGES when done
  - Constraint:      5-step limit; -0.2 penalty per wrong guess

Usage:
    # Requires a running OpenEnv-CR server on localhost:7860
    export HF_TOKEN=<your_key>          # or API_KEY
    export MODEL_NAME=gpt-4o            # optional, defaults to gpt-4o
    export API_BASE_URL=<base_url>      # optional, defaults to HF router

    python baseline_agent.py
"""

import json
import os
import sys
import textwrap
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME: str = os.getenv("MODEL_NAME") or "gpt-4o"
MAX_STEPS: int = 5
LOCAL_ENV_URL: str = "http://localhost:7860"

# ---------------------------------------------------------------------------
# System prompt  (exactly as specified in the task brief)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an Elite Security Researcher. Your task is to review a code diff provided in the Observation.

    Analysis Phase: Identify the exact line number and issue type from the allowed_issue_types.

    Action Phase: Submit your findings one by one. You must provide the issue_id and line_number exactly as they appear in the diff.

    Decision Phase: Once you have identified all bugs, or if you find no bugs, submit your final decision (APPROVE or REQUEST_CHANGES).

    Constraint: You have a strict 5-step limit. Do not hallucinate line numbers; a -0.2 penalty applies for every wrong guess.

    Output: Final answer Pass/Fail

    IMPORTANT OUTPUT FORMAT — you must respond with ONLY a single valid JSON object (no markdown, no prose):
    {
      "issue_id": "<exactly one value from allowed_issue_types, or the most relevant one>",
      "line_number": <integer — the post-diff line number of the issue>,
      "comment": "<brief explanation>",
      "decision": "APPROVE" | "REQUEST_CHANGES" | null
    }

    Rules:
    - issue_id MUST be exactly one string from the provided allowed_issue_types list.
    - line_number counts from line 1 of the diff output (including the diff header lines).
    - Set decision=null while you still have issues to report.
    - Set decision="REQUEST_CHANGES" when you are done and found issues.
    - Set decision="APPROVE" only if the diff has no issues.
    - Never repeat an issue_id + line_number pair you have already submitted (visible in history).
    - Output ONLY the JSON object — no markdown fences, no explanation.
    """
).strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check_prerequisites() -> None:
    """Validate env vars and server liveness before running."""
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable is not set.")
        sys.exit(1)

    try:
        r = requests.get(f"{LOCAL_ENV_URL}/health", timeout=5)
        r.raise_for_status()
        health = r.json()
        print(f"[OK] Server healthy — {health['num_tasks']} task(s) loaded.")
    except Exception as exc:
        print(f"ERROR: Cannot reach server at {LOCAL_ENV_URL}. Details: {exc}")
        sys.exit(1)


def parse_llm_json(raw: str) -> dict:
    """
    Extract and parse the first JSON object found in the LLM response.
    Falls back to a safe APPROVE action on parse failure to avoid crashing.
    """
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start : end + 1].strip())
    except Exception as exc:
        print(f"    [Parser Error] {exc}  |  raw={raw!r}")
    # Fallback — terminate safely without hallucinating a line number
    return {
        "issue_id": "none",
        "line_number": 0,
        "comment": "Parse failure — safe fallback.",
        "decision": "APPROVE",
    }


def build_user_prompt(obs: dict, state: dict) -> str:
    """Construct the per-step user message sent to the LLM."""
    lines = [
        f"File: {obs['file_path']}",
        "",
        "Diff:",
        obs["diff_content"],
        "",
        f"allowed_issue_types: {json.dumps(obs['allowed_issue_types'])}",
        f"Step: {state['current_step'] + 1} / {state['max_steps']}",
        f"Steps remaining: {state['max_steps'] - state['current_step']}",
    ]
    if obs.get("history"):
        lines.append("\nActions already submitted this episode:")
        for entry in obs["history"]:
            lines.append(f"  {entry}")
    lines.append(
        "\nInstruction: If you still have issues to report AND steps remain, "
        "set decision=null. Otherwise set decision=REQUEST_CHANGES or APPROVE."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(client: OpenAI, seed: int) -> float:
    """Run one full episode and return the clamped [0, 1] score."""
    print(f"\n{'='*60}")
    print(f"  Episode  |  Seed={seed}")
    print(f"{'='*60}")

    # ---- Reset ----
    res = requests.post(f"{LOCAL_ENV_URL}/reset", json={"seed": seed}, timeout=10)
    res.raise_for_status()
    data = res.json()

    obs: dict = data["observation"]
    state: dict = data["state"]
    task_id: str = data["task_id"]
    print(f"  Task: {task_id}")
    print(f"  Allowed issue types: {obs['allowed_issue_types']}")

    done = False
    total_reward = 0.0
    step_num = 0

    while not done and step_num < MAX_STEPS:
        step_num += 1
        print(f"\n  -- Step {step_num} --")

        user_prompt = build_user_prompt(obs, state)

        # ---- Call LLM ----
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw_output: str = response.choices[0].message.content or "{}"
            print(f"    LLM raw: {raw_output.strip()}")
            action_dict = parse_llm_json(raw_output)
        except Exception as exc:
            print(f"    [LLM Error] {exc}")
            action_dict = {
                "issue_id": "none",
                "line_number": 0,
                "comment": "LLM call failed.",
                "decision": "APPROVE",
            }

        print(f"    Submitting: {json.dumps(action_dict)}")

        # ---- Submit action ----
        step_res = requests.post(
            f"{LOCAL_ENV_URL}/step", json=action_dict, timeout=10
        )
        if step_res.status_code != 200:
            print(f"    [Server Error {step_res.status_code}] {step_res.text}")
            break

        step_data = step_res.json()
        reward: float = step_data["reward"]
        done = step_data["done"]
        obs = step_data["observation"]
        state = step_data["state"]
        info = step_data.get("info", {})

        total_reward += reward
        print(
            f"    reward={reward:+.3f}  |  total_so_far={total_reward:+.3f}  "
            f"|  done={done}  |  reason={info.get('reason', '')}"
        )

        # Small delay to stay within rate limits when hitting cloud APIs
        if not done:
            time.sleep(0.3)

    # Clamp to [0, 1] as defined by the hackathon scoring spec
    clamped = max(0.0, min(1.0, total_reward))
    print(f"\n  Episode complete  |  raw_total={total_reward:.3f}  |  score={clamped:.3f}")
    return clamped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    check_prerequisites()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Discover how many tasks the server has
    health_info = requests.get(f"{LOCAL_ENV_URL}/health", timeout=5).json()
    num_tasks: int = health_info.get("num_tasks", 3)

    scores: list[float] = []
    for seed in range(num_tasks):
        score = run_episode(client, seed)
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    print("\n" + "=" * 60)
    print("  BASELINE AGENT — FINAL RESULTS")
    print("=" * 60)
    for i, s in enumerate(scores):
        verdict = "PASS" if s >= 0.5 else "FAIL"
        print(f"  Seed {i}: score={s:.3f}  [{verdict}]")
    print(f"\n  AVERAGE SCORE : {avg_score:.3f}")
    overall = "Pass" if avg_score >= 0.5 else "Fail"
    print(f"  FINAL ANSWER  : {overall}")
    print("=" * 60)


if __name__ == "__main__":
    main()
