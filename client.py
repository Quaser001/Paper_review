"""
client.py — Thin OpenEnv client for peer_review_env.

Usage:
    import asyncio
    from peer_review_env.client import PeerReviewEnvClient, PeerReviewAction

    async def main():
        async with PeerReviewEnvClient(base_url="http://localhost:8000") as env:
            obs = await env.reset()
            result = await env.step(PeerReviewAction(
                recommendation="reject",
                identified_flaws=["p-value threshold was adjusted post-hoc"],
                confidence=0.9,
                reasoning="Methodology section reveals iterative seed selection."
            ))

    asyncio.run(main())
"""
from openenv.core import EnvClient

try:
    from .models import PeerReviewAction, PeerReviewObservation, PeerReviewState
except ImportError:
    from models import PeerReviewAction, PeerReviewObservation, PeerReviewState

from typing import Any, Dict
from openenv.core.client_types import StepResult


class PeerReviewEnvClient(EnvClient[PeerReviewAction, PeerReviewObservation, PeerReviewState]):
    """Typed async client for peer_review_env."""

    def _step_payload(self, action: PeerReviewAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PeerReviewObservation]:
        """Parse the server response into a StepResult.

        The server sends: { "observation": {...}, "reward": float, "done": bool }
        via serialize_observation() in openenv.
        """
        obs_data = payload.get("observation", {})
        reward = payload.get("reward")
        done = payload.get("done", False)

        obs = PeerReviewObservation.model_validate(obs_data)

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PeerReviewState:
        return PeerReviewState.model_validate(payload)
