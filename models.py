"""
Pydantic data contracts for peer_review_env.
All agent-environment communication is type-safe.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Agent Action
# ---------------------------------------------------------------------------

class PeerReviewAction(Action):
    """Structured peer review output from the agent."""

    recommendation: Literal["accept", "minor_revision", "major_revision", "reject"] = Field(
        ...,
        description=(
            "Editorial decision. Ordinal scale: "
            "accept > minor_revision > major_revision > reject"
        ),
    )
    identified_flaws: List[str] = Field(
        default_factory=list,
        description=(
            "List of natural-language flaw descriptions found in the paper. "
            "Be specific. Empty list for a paper you believe is flawless."
        ),
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent self-reported confidence in this review (0.0-1.0).",
    )
    reasoning: str = Field(
        default="",
        description="Chain-of-thought justification for the recommendation and identified flaws.",
    )


# ---------------------------------------------------------------------------
# Environment Observation (returned to agent)
# ---------------------------------------------------------------------------

class PaperContent(BaseModel):
    """The manuscript content presented to the agent."""

    paper_id: str
    title: str
    abstract: str
    methodology: str
    results: str
    conclusions: str
    citations: List[str]


class PeerReviewObservation(Observation):
    """What the agent sees after reset() or step().

    Inherits from openenv Observation which provides:
      - done: bool (episode termination flag)
      - reward: float | None (reward signal)
      - metadata: Dict[str, Any]
    """

    paper: Optional[PaperContent] = Field(
        None,
        description="The paper to review. Populated after reset().",
    )
    feedback: str = Field(
        "",
        description="Human-readable feedback on the last action for debugging.",
    )
    task_name: str = Field(
        "",
        description="Task difficulty level: easy | medium | hard",
    )


# ---------------------------------------------------------------------------
# Environment State (internal bookkeeping exposed via /state)
# ---------------------------------------------------------------------------

class PeerReviewState(State):
    """Internal episode state for trainers and evaluators."""

    task_name: str = Field("", description="Current task: easy | medium | hard")
    paper_id: str = Field("", description="ID of paper being reviewed.")
    done: bool = Field(False, description="Episode termination flag.")
    curriculum_level: float = Field(
        1.0,
        ge=1.0,
        description="Dynamic difficulty scalar; increases as agent improves.",
    )
    agent_rolling_score: float = Field(
        0.0,
        description="Rolling average reward over last 5 episodes.",
    )
    last_reward: float = Field(0.0, description="Reward from most recent step.")
