"""
peer_review_environment.py — Core environment state machine.

3 tasks (easy -> medium -> hard):
  easy   : correct papers (no flaws)
  medium : p_hacking, no_cross_validation, overclaimed_results
  hard   : fabricated_citation, impossible_statistics (critical, w=1.0)

Curriculum: agent_rolling_score tracked over last 5 episodes.
If rolling_score > 0.75 for 3 consecutive episodes twice, curriculum_level increases.
"""
from __future__ import annotations

import json
import logging
import random
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

try:
    from ..models import (
        PaperContent,
        PeerReviewAction,
        PeerReviewObservation,
        PeerReviewState,
    )
    from ..reward import compute_reward
except ImportError:
    from models import (
        PaperContent,
        PeerReviewAction,
        PeerReviewObservation,
        PeerReviewState,
    )
    from reward import compute_reward

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASK_FLAW_TYPES = {
    "easy": ["correct_paper_no_flaw"],
    "medium": ["p_hacking", "no_cross_validation", "overclaimed_results"],
    "hard": ["fabricated_citation", "impossible_statistics"],
}

DATA_PATH = Path(__file__).parent.parent / "data" / "papers.json"


def _load_papers() -> List[Dict]:
    with open(DATA_PATH, "r") as f:
        return json.load(f)


class PeerReviewEnvironment(Environment):
    """
    OpenEnv-compliant environment for automated peer review.

    Episodes are single-turn (contextual bandit): agent receives a paper,
    outputs a review, receives reward, episode terminates.
    """

    def __init__(self) -> None:
        super().__init__()
        self._papers = _load_papers()
        self._paper_by_type: Dict[str, List[Dict]] = {}
        for p in self._papers:
            ft = p["flaw_type"]
            self._paper_by_type.setdefault(ft, []).append(p)

        self._episode_id: str = ""
        self._current_paper: Optional[Dict] = None
        self._task_name: str = "easy"
        self._step_count: int = 0
        self._done: bool = True

        # Curriculum tracking
        self._curriculum_level: float = 1.0
        self._score_history: deque = deque(maxlen=5)
        self._high_score_streak: int = 0
        self._rolling_score: float = 0.0
        self._last_reward: float = 0.0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_name: Optional[str] = None, **kwargs: Any) -> PeerReviewObservation:
        """
        Start a new episode. Selects a paper matching the requested task.
        If task_name is None, auto-selects based on curriculum_level.
        """
        if seed is not None:
            random.seed(seed)

        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._done = False

        # Determine task
        if task_name is None:
            task_name = self._curriculum_task()
        self._task_name = task_name

        # Pick paper
        self._current_paper = self._sample_paper(task_name)

        paper_content = PaperContent(
            paper_id=self._current_paper["paper_id"],
            title=self._current_paper["title"],
            abstract=self._current_paper["abstract"],
            methodology=self._current_paper["methodology"],
            results=self._current_paper["results"],
            conclusions=self._current_paper["conclusions"],
            citations=self._current_paper["citations"],
        )

        return PeerReviewObservation(
            paper=paper_content,
            reward=None,
            feedback="New paper loaded. Submit your peer review.",
            task_name=task_name,
            done=False,
        )

    def step(self, action: PeerReviewAction, timeout_s: Optional[float] = None,
             **kwargs: Any) -> PeerReviewObservation:
        """
        Process a peer review action. Returns observation with reward.
        Episode terminates after one step (contextual bandit).
        """
        if self._done or self._current_paper is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1
        self._done = True  # single-turn episode

        reward_info = compute_reward(
            predicted_recommendation=action.recommendation,
            identified_flaws=action.identified_flaws,
            confidence=action.confidence,
            ground_truth_recommendation=self._current_paper["ground_truth_review"],
            ground_truth_flaws=self._current_paper["correct_flaws_list"],
            flaw_type=self._current_paper["flaw_type"],
        )

        r = reward_info.r_total
        self._last_reward = r

        # Update curriculum tracking
        self._score_history.append(r)
        self._rolling_score = sum(self._score_history) / len(self._score_history)
        self._update_curriculum()

        feedback = self._build_feedback(action, reward_info)

        return PeerReviewObservation(
            paper=None,
            reward=r,
            feedback=feedback,
            task_name=self._task_name,
            done=True,
            metadata={
                "r_flaw": reward_info.r_flaw,
                "a_rec": reward_info.a_rec,
                "b_crit": reward_info.b_crit,
                "p_fp": reward_info.p_fp,
                "gate": reward_info.gate,
                "tp": reward_info.tp,
                "fp": reward_info.fp,
                "fn": reward_info.fn,
                "details": reward_info.details,
                "ground_truth_recommendation": self._current_paper["ground_truth_review"],
                "ground_truth_flaws": self._current_paper["correct_flaws_list"],
                "flaw_type": self._current_paper["flaw_type"],
                "curriculum_level": self._curriculum_level,
            },
        )

    @property
    def state(self) -> PeerReviewState:
        return PeerReviewState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            paper_id=self._current_paper["paper_id"] if self._current_paper else "",
            done=self._done,
            curriculum_level=self._curriculum_level,
            agent_rolling_score=self._rolling_score,
            last_reward=self._last_reward,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _curriculum_task(self) -> str:
        """Map curriculum_level to a task name."""
        if self._curriculum_level < 1.5:
            return "easy"
        elif self._curriculum_level < 2.5:
            return random.choice(["easy", "medium"])
        else:
            return random.choice(["medium", "hard"])

    def _update_curriculum(self) -> None:
        """Increase difficulty if agent consistently scores high."""
        if len(self._score_history) < 3:
            return
        recent = list(self._score_history)[-3:]
        if all(s > 0.75 for s in recent):
            self._high_score_streak += 1
            if self._high_score_streak >= 2:
                self._curriculum_level = min(self._curriculum_level + 0.5, 3.0)
                self._high_score_streak = 0
        else:
            self._high_score_streak = 0

    def _sample_paper(self, task_name: str) -> Dict:
        """Pick a random paper matching the task difficulty."""
        flaw_types = TASK_FLAW_TYPES[task_name]
        candidates = []
        for ft in flaw_types:
            candidates.extend(self._paper_by_type.get(ft, []))
        if not candidates:
            raise ValueError(f"No papers found for task={task_name}")
        return random.choice(candidates)

    def _build_feedback(self, action: PeerReviewAction, reward_info: Any) -> str:
        gt_rec = self._current_paper["ground_truth_review"]
        gt_flaws = self._current_paper["correct_flaws_list"]
        lines = [
            f"Reward: {reward_info.r_total:.3f}",
            f"Your recommendation: {action.recommendation} | Correct: {gt_rec}",
            f"Flaws found: {reward_info.tp}/{len(gt_flaws)} "
            f"(FP={reward_info.fp}, FN={reward_info.fn})",
            f"Gate: {'OPEN' if reward_info.gate else 'CLOSED (missed too many flaws)'}",
            reward_info.details,
        ]
        if gt_flaws:
            lines.append(f"Ground truth flaws: {'; '.join(gt_flaws)}")
        return " | ".join(lines)
