"""
graders.py — Task graders for peer_review_env.

Each grader runs N episodes of a specific difficulty, returns mean reward.
Used by the hackathon automated evaluator.

Usage:
    python graders.py --task easy --episodes 5
    python graders.py --task medium --episodes 5
    python graders.py --task hard --episodes 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import os

from client import PeerReviewEnvClient, PeerReviewAction

logger = logging.getLogger(__name__)

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")


async def grade_task(task_name: str, n_episodes: int = 5) -> dict:
    """
    Run n_episodes of a task against a naive baseline agent.
    Baseline: always 'reject', no identified flaws (tests reward function floor).
    Returns score summary with min/max/mean in [0, 1].
    """
    scores = []

    async with PeerReviewEnvClient(base_url=ENV_BASE_URL) as env:
        for ep in range(n_episodes):
            logger.info("Running episode %d/%d for task '%s'...", ep + 1, n_episodes, task_name)

            # Reset environment
            reset_result = await env.reset(task_name=task_name)

            # Naive action
            action = PeerReviewAction(
                recommendation="reject",
                identified_flaws=[],
                confidence=0.5,
                reasoning="Baseline grader: conservative reject.",
            )

            step_result = await env.step(action)
            reward = step_result.reward if step_result.reward is not None else 0.0

            # Clamp to [0, 1] to be safe
            reward = max(0.0, min(1.0, reward))
            scores.append(reward)

    return {
        "task": task_name,
        "episodes": n_episodes,
        "scores": scores,
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description="Peer review env grader")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    result = asyncio.run(grade_task(args.task, args.episodes))
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
