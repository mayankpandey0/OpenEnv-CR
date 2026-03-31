---
title: OpenEnv CR
emoji: "🚀"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv-CR -- Code Review Simulation Environment

A **fully deterministic, reproducible** OpenEnv-compatible environment where an AI agent reviews unified git diffs and is evaluated via a strict programmatic grading system -- no LLMs involved in grading.

---

## Architecture

```
                     Agent Layer
  +---------------------------------------------------+
  |               inference.py (LLM Agent)            |
  +---------------------------------------------------+
               |
               |  HTTP: /reset, /step
               +-----------+--------------+
                           |
                           v
  +---------------------------------------------------+
  |           server/env.py  (FastAPI)                |
  |  POST /reset  -> select task via seed % num_tasks  |
  |  POST /step   -> validate -> grade -> mutate state |
  |  GET  /state  -> read-only snapshot                |
  |  GET  /observation -> diff + allowed types         |
  |  GET  /health -> liveness probe                    |
  +---------------------------------------------------+
                           |
                    pure function call
                           |
                           v
  +---------------------------------------------------+
  |                  server/logic.py                   |
  |  compute_reward(action, state, ground_truth, ...)  |
  +---------------------------------------------------+
                           |
                           v
  +---------------------------------------------------+
  |                server/tasks.json                   |
  |  3 tasks: easy / medium / hard                     |
  +---------------------------------------------------+
```

---

## Project Structure

```
OpenEnv-CR/
|- server/
|  |- models.py             # Pydantic schemas: Action, Observation, State
|  |- tasks.json            # Task database (easy / medium / hard)
|  |- logic.py              # Pure reward grader
|  +- env.py                # FastAPI server
|- inference.py             # Example LLM agent using OpenAI client
|- push.py                  # Script to upload environment as a Hugging Face Space
|- openenv.yaml             # OpenEnv metadata
|- pyproject.toml           # Project metadata
|- requirements.txt         # Pinned Python dependencies
|- Dockerfile               # python:3.11-slim container with HEALTHCHECK
|- Makefile                 # Developer shortcuts
|- tests/
|  |- test_logic.py         # Unit tests for the grader
|  +- test_env.py           # Integration tests via TestClient
+- README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
make serve

# 3. Run all unit + integration tests (no server needed)
make test-all

# 4. (Optional) Run the sample LLM agent
export API_KEY="sk-..."
make test-llm
```

---

## Agents

### Example LLM Agent (`inference.py`)

`inference.py` -- uses the OpenAI client for code review reasoning. By default uses `gpt-4o`.

- Requires `HF_TOKEN` (or `API_KEY`) environment variable
- Can use custom `API_BASE_URL` (default: Hugging Face router)
- Uses `temperature=0.0` for near-deterministic output

**Run:**
```bash
export API_KEY="sk-..."
python inference.py
```

---

## API Reference

### `POST /reset`

Start a new episode. Deterministically selects a task via `seed % num_tasks`.

**Request:**
```json
{ "seed": 0 }
```

**Response:**
```json
{
  "task_id": "easy_syntax_1",
  "seed": 0,
  "observation": {
    "diff_content": "--- a/greet.py\n+++ b/greet.py\n...",
    "file_path": "greet.py",
    "allowed_issue_types": ["missing_colon", "variable_not_defined", "unused_import"],
    "history": []
  },
  "state": {
    "current_step": 0,
    "max_steps": 5,
    "task_id": "easy_syntax_1",
    "found_issue_ids": [],
    "done": false
  }
}
```

### `POST /step`

Submit one agent action. Returns reward, diagnostic info, and updated state.

**Request:**
```json
{
  "issue_id": "missing_colon",
  "line_number": 2,
  "comment": "if-statement is missing a colon at the end.",
  "decision": null
}
```

**Response:**
```json
{
  "reward": 0.5,
  "info": {
    "valid_hit": true,
    "hallucination": false,
    "duplicate": false,
    "correct_decision": false,
    "reason": "Valid hit: line 2 -> 'missing_colon'."
  },
  "done": false,
  "observation": { "...": "..." },
  "state": { "current_step": 1, "done": false, "...": "..." }
}
```

To end the episode, include `"decision": "REQUEST_CHANGES"` or `"decision": "APPROVE"`.

### `GET /state`

Returns the current state snapshot (read-only).

### `GET /observation`

Returns the current observation (diff content, allowed issue types, history).

### `GET /health`

Liveness probe. Returns `{"status": "ok", "num_tasks": 3}`.

---

## Reward Formula

```
R = ((0.7 / num_ground_truth_issues) * ValidHit) - (0.2 * Hallucination) + (0.3 * CorrectDecision)
```

| Component         | Condition                                                         | Value |
|-------------------|-------------------------------------------------------------------|-------|
| `ValidHit`        | `line_number` in GT AND `issue_id` matches AND not dup            | `0.7/N`|
| `Hallucination`   | `line_number` not in GT OR `issue_id` mismatch OR OOV             | -0.2  |
| `Duplicate`       | `issue_id` already found this episode                             | 0.0   |
| `CorrectDecision` | Terminal `decision` matches task's `correct_decision`             | +0.3  |

**Max reward per episode:** `1.0`

---

## Anti-Exploitation Rules

| Rule                   | Enforcement                                                   |
|------------------------|---------------------------------------------------------------|
| `max_steps = 5`        | Episode ends when `current_step >= max_steps`                 |
| OOV `issue_id`         | Hallucination penalty (-0.2)                                  |
| Duplicate `issue_id`   | Zero reward (no exploit by re-submitting)                     |
| Decision ends episode  | No more steps allowed after `done=True`                       |
| Random guessing        | Hallucinations accumulate, making guessing unprofitable       |

---

## Task Database

| ID                   | Difficulty | Unique Issues | Description                                                 |
|----------------------|------------|---------------|-------------------------------------------------------------|
| `easy_syntax_1`      | Easy       | 1*            | Missing colons in if/elif statements                        |
| `medium_resource_1`  | Medium     | 2             | Unclosed file handle + unguarded index access               |
| `hard_idor_1`        | Hard       | 2             | IDOR vulnerability + missing email validation in FastAPI    |

*Easy task has 2 ground truth entries with the same `issue_id` ("missing_colon"), so only 1 unique reward is possible.

---

## Docker

```bash
docker build -t openenv-cr .
docker run -p 7860:7860 openenv-cr
```

Container includes a `HEALTHCHECK` on `/health`.

---

## Running Tests

```bash
# Unit tests only (no server needed, no API key needed)
pytest tests/test_logic.py -v

# Integration tests (uses FastAPI TestClient -- no live server)
pytest tests/test_env.py -v

# All tests
pytest tests/ -v
```

---

## Global Constraints

- **Fully deterministic**: `reset(seed=N)` always selects the same task
- **No LLM grading**: all rewards computed by pure Python logic
- **No reward exploitation**: duplicate hits, OOV IDs, and random guessing all penalised
- **Reproducible**: fixed `tasks.json`, pinned `requirements.txt`
