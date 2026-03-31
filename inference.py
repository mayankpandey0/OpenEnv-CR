"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import sys
import json
import time
import requests
import textwrap

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o"
MAX_STEPS = 5

LOCAL_ENV_URL = "http://localhost:7860"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a deterministic AI code reviewer.
    Analyze the provided unified diff and identify real code issues.

    You MUST output ONLY a valid JSON object matching this exact schema:
    {
      "issue_id": "<one of the allowed_issue_types list>",
      "line_number": <post-diff integer line number>,
      "comment": "<brief explanation of the issue>",
      "decision": "APPROVE" | "REQUEST_CHANGES" | null
    }

    Rules:
    - issue_id MUST be exactly one value from allowed_issue_types, or null if APPROVE.
    - line_number is the post-diff line number where the issue appears.
    - Set decision=REQUEST_CHANGES or decision=APPROVE only when you are done.
    - While you still have issues to report, set decision=null.
    - Do NOT output markdown, code fences, explanations, or chain-of-thought.
    - Output ONLY the JSON object.
    """
).strip()


def check_prerequisites():
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable is not set.")
        sys.exit(1)
        
    try:
        r = requests.get(f"{LOCAL_ENV_URL}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach server at {LOCAL_ENV_URL}. Details: {e}")
        sys.exit(1)


def parse_llm_json(text: str) -> dict:
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1].strip())
    except Exception as e:
        print(f"    [Parser Error] Could not parse JSON. Raw: {text!r}")
    return {"issue_id": "none", "line_number": 0, "comment": "fallback", "decision": "APPROVE"}


def run_episode(client: OpenAI, seed: int) -> float:
    print(f"\n--- Starting Episode (Seed: {seed}) ---")
    res = requests.post(f"{LOCAL_ENV_URL}/reset", json={"seed": seed}, timeout=10)
    res.raise_for_status()
    data = res.json()
    
    obs = data["observation"]
    state = data["state"]
    done = False
    total_reward = 0.0

    step_num = 0
    while not done and step_num < MAX_STEPS:
        step_num += 1

        user_prompt = (
            f"File: {obs['file_path']}\n\n"
            f"Diff:\n{obs['diff_content']}\n\n"
            f"Allowed issue types: {json.dumps(obs['allowed_issue_types'])}\n"
            f"Current step: {state['current_step'] + 1}/{state['max_steps']}\n"
        )
        if obs.get("history"):
            user_prompt += f"\nPrevious actions:\n" + "\n".join(f"  {h}" for h in obs["history"]) + "\n"

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1000
            )
            raw_output = response.choices[0].message.content or "{}"
            action_dict = parse_llm_json(raw_output)
        except Exception as e:
            print(f"  Step {step_num}: LLM generation failed ({e})")
            action_dict = {"issue_id": "none", "line_number": 0, "comment": "error", "decision": "APPROVE"}

        print(f"  Step {step_num}: submit -> {json.dumps(action_dict)}")
        
        step_res = requests.post(f"{LOCAL_ENV_URL}/step", json=action_dict, timeout=10)
        if step_res.status_code != 200:
            print(f"  Step {step_num}: server error -> {step_res.text}")
            break

        step_data = step_res.json()
        reward = step_data["reward"]
        done = step_data["done"]
        obs = step_data["observation"]
        state = step_data["state"]
        
        total_reward += reward

    # Clean bounds mapping exactly to hackathon limit
    total_reward = max(0.0, min(1.0, total_reward))
    print(f"  Episode Complete | Score: {total_reward:.2f}")
    return total_reward


def main():
    check_prerequisites()
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Discover tasks dynamically
    health_info = requests.get(f"{LOCAL_ENV_URL}/health", timeout=5).json()
    num_tasks = health_info.get("num_tasks", 3)

    overall_score = 0.0
    for seed in range(num_tasks):
        score = run_episode(client, seed)
        overall_score += score

    avg_score = overall_score / num_tasks
    print(f"\n==========================================")
    print(f"  FINAL AVERAGE SCORE: {avg_score:.2f} (Over {num_tasks} tasks)")
    print(f"==========================================")


if __name__ == "__main__":
    main()
