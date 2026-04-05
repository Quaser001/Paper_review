"""
inference.py — Baseline inference script for peer_review_env.

Runs 3 tasks (easy, medium, hard) using an LLM agent via OpenAI-compatible API.
Emits strictly formatted [START], [STEP], [END] stdout logs for the evaluator.

Environment variables required:
    API_BASE_URL  — LLM API endpoint (default: https://api-inference.huggingface.co/v1)
    MODEL_NAME    — model identifier (default: Qwen/Qwen2.5-1.5B-Instruct)
    HF_TOKEN      — Hugging Face / API key
    ENV_BASE_URL  — environment server URL (default: http://localhost:7860)

Run:
    python inference.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict

from openai import OpenAI
from client import PeerReviewEnvClient, PeerReviewAction

# ---------------------------------------------------------------------------
# Logging — stderr only, never pollute stdout [START][STEP][END] stream
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment (MANDATORY per competition rules)
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

llm_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL) if HF_TOKEN else None

# ---------------------------------------------------------------------------
# Stdout log helpers (EXACT format required by evaluator)
# ---------------------------------------------------------------------------

def _emit_start(task_name: str, paper_id: str, episode_id: str) -> None:
    record = {
        "task_name": task_name,
        "paper_id": paper_id,
        "episode_id": episode_id,
        "timestamp": time.time(),
    }
    sys.stdout.write(f"[START] {json.dumps(record)}\n")
    sys.stdout.flush()


def _emit_step(
    step: int,
    action: Dict,
    observation: Dict,
    reward: float,
    done: bool,
) -> None:
    record = {
        "step": step,
        "action": action,
        "observation": observation,
        "reward": reward,
        "done": done,
        "timestamp": time.time(),
    }
    sys.stdout.write(f"[STEP] {json.dumps(record)}\n")
    sys.stdout.flush()


def _emit_end(task_name: str, total_reward: float, steps: int, success: bool) -> None:
    record = {
        "task_name": task_name,
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "timestamp": time.time(),
    }
    sys.stdout.write(f"[END] {json.dumps(record)}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert scientific peer reviewer with deep expertise in
machine learning and reinforcement learning methodology.

You will be given a scientific paper (title, abstract, methodology, results, conclusions,
and citations). Your task is to produce a structured peer review.

You MUST respond with valid JSON ONLY — no preamble, no markdown, no explanation outside the JSON.

JSON schema:
{
  "recommendation": "<accept|minor_revision|major_revision|reject>",
  "identified_flaws": ["<specific flaw description>", ...],
  "confidence": <float 0.0-1.0>,
  "reasoning": "<chain-of-thought explanation>"
}

Common flaws to check:
1. P-hacking / selective seed reporting (look for post-hoc threshold adjustments, cherry-picked seeds)
2. No cross-validation / evaluation on training data (same environment config for train and test)
3. Overclaimed results (abstract claims far exceed evidence in results section)
4. Fabricated citations (non-existent papers, fake DOIs, future conference dates)
5. Impossible statistics (Cohen's d > 3 in small N; GRIM test failures for integer means)
6. Methodological deviations without justification

For the GRIM test: if N is small and the metric is integer-based, check that the reported
mean is a multiple of 1/N. If not, flag as impossible statistic.

Be precise. Name the exact location of each flaw (section + specific text).
Empty identified_flaws list if the paper is methodologically sound."""


def llm_review(paper: Dict) -> Dict:
    """Call LLM to produce a structured peer review using OpenAI client."""
    if llm_client is None:
        logger.warning("No HF_TOKEN set, using fallback review")
        return {
            "recommendation": "reject",
            "identified_flaws": [],
            "confidence": 0.1,
            "reasoning": "No API token available.",
        }

    citations_text = "\n".join(f"- {c}" for c in paper.get("citations", []))
    paper_text = (
        f"Title: {paper.get('title', '')}\n\n"
        f"Abstract:\n{paper.get('abstract', '')}\n\n"
        f"Methodology:\n{paper.get('methodology', '')}\n\n"
        f"Results:\n{paper.get('results', '')}\n\n"
        f"Conclusions:\n{paper.get('conclusions', '')}\n\n"
        f"Citations:\n{citations_text}"
    )

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Please review this paper:\n\n{paper_text}"},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("LLM returned invalid JSON: %s", exc)
        return {
            "recommendation": "reject",
            "identified_flaws": [],
            "confidence": 0.1,
            "reasoning": "Failed to parse LLM response.",
        }
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return {
            "recommendation": "major_revision",
            "identified_flaws": [],
            "confidence": 0.1,
            "reasoning": f"API error: {exc}",
        }


# ---------------------------------------------------------------------------
# Sanitize LLM output to match Pydantic schema
# ---------------------------------------------------------------------------
VALID_RECS = {"accept", "minor_revision", "major_revision", "reject"}

def _sanitize_recommendation(raw: str) -> str:
    """Normalize LLM recommendation to valid Literal value."""
    r = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if r in VALID_RECS:
        return r
    if "accept" in r:
        return "accept"
    if "minor" in r:
        return "minor_revision"
    if "major" in r:
        return "major_revision"
    return "reject"


def _sanitize_review(review: Dict) -> Dict:
    """Ensure LLM output conforms to PeerReviewAction schema."""
    rec = _sanitize_recommendation(str(review.get("recommendation", "reject")))

    flaws = review.get("identified_flaws", [])
    if not isinstance(flaws, list):
        flaws = []
    flaws = [str(f) for f in flaws if f]

    try:
        conf = float(review.get("confidence", 0.5))
    except (ValueError, TypeError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    reasoning = str(review.get("reasoning", ""))

    return {
        "recommendation": rec,
        "identified_flaws": flaws,
        "confidence": conf,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Main loop — 3 tasks
# ---------------------------------------------------------------------------

async def run_task(task_name: str, env: PeerReviewEnvClient) -> float:
    """Run a single episode for the given task."""
    logger.info("Starting task: %s", task_name)

    episode_id = str(uuid.uuid4())

    reset_result = await env.reset(task_name=task_name)
    obs = reset_result.observation
    paper_obj = obs.paper if obs else None

    if paper_obj and hasattr(paper_obj, "model_dump"):
        paper = paper_obj.model_dump()
    elif isinstance(paper_obj, dict):
        paper = paper_obj
    else:
        paper = {}

    paper_id = paper.get("paper_id", "unknown")
    _emit_start(task_name, paper_id, episode_id)

    review = llm_review(paper)
    sanitized = _sanitize_review(review)

    action = PeerReviewAction(
        recommendation=sanitized["recommendation"],
        identified_flaws=sanitized["identified_flaws"],
        confidence=sanitized["confidence"],
        reasoning=sanitized["reasoning"],
    )

    step_result = await env.step(action)
    reward = step_result.reward if step_result.reward is not None else 0.0
    done = step_result.done
    step_obs = step_result.observation

    _emit_step(
        step=1,
        action=action.model_dump(),
        observation={
            "feedback": getattr(step_obs, "feedback", ""),
            "task_name": getattr(step_obs, "task_name", task_name),
            "done": done,
        },
        reward=reward,
        done=done,
    )

    success = reward >= 0.5
    _emit_end(task_name, total_reward=reward, steps=1, success=success)

    logger.info("Task %s complete. Reward=%.3f Success=%s", task_name, reward, success)
    return reward


async def amain() -> None:
    tasks = ["easy", "medium", "hard"]
    scores: Dict[str, float] = {}

    async with PeerReviewEnvClient(base_url=ENV_BASE_URL) as env:
        for task in tasks:
            try:
                scores[task] = await run_task(task, env)
            except Exception as exc:
                logger.error("Task %s failed: %s", task, exc)
                _emit_end(task, total_reward=0.0, steps=0, success=False)
                scores[task] = 0.0

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    logger.info("All tasks complete. Scores: %s. Average: %.3f", scores, avg)


def main() -> None:
    try:
        asyncio.run(amain())
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
    sys.exit(0)


if __name__ == "__main__":
    main()
