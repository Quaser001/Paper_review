"""
reward.py — Full mathematical reward function implementation.

Based on: "Mathematical Formulation of a Verifiable Reward Function
for Reinforcement Learning in Automated Peer Review"

R_total = max(0, min(1, R_raw / R_max_possible))

R_raw = Γ·(α·A_rec) + β·R_flaw + B_crit - P_FP

Where:
  Γ     = gating function (1 if R_flaw >= γ, else 0)
  A_rec = severity-weighted recommendation accuracy
  R_flaw= severity-weighted recall via SBERT + Hungarian matching
  B_crit= bonus for detecting critical flaws (w=1.0)
  P_FP  = hallucination penalty

SBERT model: all-MiniLM-L6-v2 (22MB, CPU-fast, runs in <1s per paper)
Matching   : scipy linear_sum_assignment (Hungarian algorithm)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy imports — loaded once at first reward computation
_sbert_model = None

# ---------------------------------------------------------------------------
# Hyperparameters (from worked examples in the PDF)
# ---------------------------------------------------------------------------
ALPHA = 0.4          # recommendation weight
BETA = 0.6           # flaw recall weight
LAMBDA_FP = 0.15     # false-positive penalty per hallucination
LAMBDA_BONUS = 0.3   # critical flaw bonus magnitude
GAMMA = 0.4          # gating threshold (40% severity-weighted recall)
KAPPA = 1.5          # severity exponential for leniency penalty
TAU = 0.75           # SBERT cosine similarity threshold

# Ordinal scale for recommendations
REC_ORDINAL = {
    "reject": 0,
    "major_revision": 1,
    "minor_revision": 2,
    "accept": 3,
}

# Flaw severity weights
SEVERITY_WEIGHTS = {
    "p_hacking": 0.8,
    "no_cross_validation": 0.7,
    "overclaimed_results": 0.5,
    "fabricated_citation": 1.0,
    "impossible_statistics": 1.0,
    "correct_paper_no_flaw": 0.0,
    # Generic fallback for custom flaws
    "formatting": 0.1,
    "scope": 0.3,
    "over_interpretation": 0.5,
    "methodological_deviation": 0.6,
    "unsupported_conclusions": 0.7,
    "flawed_study_design": 0.8,
}


# ---------------------------------------------------------------------------
# SBERT loader (lazy, cached)
# ---------------------------------------------------------------------------

def _get_sbert():
    global _sbert_model
    if _sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError(
                f"sentence-transformers not installed or model unavailable: {e}"
            )
    return _sbert_model


# ---------------------------------------------------------------------------
# Cosine similarity matrix
# ---------------------------------------------------------------------------

def _cosine_sim_matrix(predicted: List[str], ground_truth: List[str]) -> np.ndarray:
    """Build n×m cosine similarity matrix via SBERT embeddings."""
    model = _get_sbert()
    all_texts = predicted + ground_truth
    embeddings = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
    pred_emb = embeddings[: len(predicted)]
    gt_emb = embeddings[len(predicted):]
    # dot product of unit vectors = cosine similarity
    sim = pred_emb @ gt_emb.T  # shape (n, m)
    return sim


# ---------------------------------------------------------------------------
# Hungarian algorithm matching
# ---------------------------------------------------------------------------

def _hungarian_match(
    predicted: List[str],
    ground_truth: List[str],
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Maximum weight bipartite matching (Linear Sum Assignment).
    Returns matched pairs (i, j) and the full similarity matrix.
    Only pairs where S_ij >= TAU are valid.
    """
    if not predicted or not ground_truth:
        return [], np.zeros((len(predicted), len(ground_truth)))

    sim = _cosine_sim_matrix(predicted, ground_truth)

    from scipy.optimize import linear_sum_assignment
    # Negate for minimization
    row_ind, col_ind = linear_sum_assignment(-sim)

    matched = [
        (r, c)
        for r, c in zip(row_ind, col_ind)
        if sim[r, c] >= TAU
    ]
    return matched, sim


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    r_flaw: float
    a_rec: float
    b_crit: float
    p_fp: float
    gate: float
    r_raw: float
    r_total: float
    tp: int
    fp: int
    fn: int
    details: str


def compute_reward(
    predicted_recommendation: str,
    identified_flaws: List[str],
    confidence: float,
    ground_truth_recommendation: str,
    ground_truth_flaws: List[str],
    flaw_type: str,
) -> RewardBreakdown:
    """
    Compute R_total ∈ [0, 1] given agent output and oracle ground truth.

    For correct papers (no flaws), R_flaw=1.0 automatically, gate=1.
    Penalty still active: hallucinating flaws on a correct paper = FP.
    """
    # -----------------------------------------------------------------------
    # Get severity weights for each ground-truth flaw
    # -----------------------------------------------------------------------
    w_base = SEVERITY_WEIGHTS.get(flaw_type, 0.5)

    # Each canonical flaw gets the base weight for this paper type
    # (in production you'd have per-flaw weights in the oracle)
    gt_weights = [w_base] * len(ground_truth_flaws)
    w_sum = sum(gt_weights) if gt_weights else 1.0
    w_max = max(gt_weights) if gt_weights else 0.0

    # -----------------------------------------------------------------------
    # Perfect paper edge case
    # -----------------------------------------------------------------------
    if not ground_truth_flaws:
        # R_flaw = 1.0, gate = 1, no bonus possible
        r_flaw = 1.0
        gate = 1.0
        tp, fn = 0, 0
        fp = len(identified_flaws)
        # No matching needed; all predictions are FP
        matched_pairs = []
        sim_matrix = np.zeros((fp, 0))
    else:
        # -----------------------------------------------------------------------
        # Hungarian matching
        # -----------------------------------------------------------------------
        matched_pairs, sim_matrix = _hungarian_match(identified_flaws, ground_truth_flaws)
        tp = len(matched_pairs)
        fp = len(identified_flaws) - tp
        fn = len(ground_truth_flaws) - tp

        # -----------------------------------------------------------------------
        # Constraint 1: Severity-Weighted Recall
        # R_flaw = Σ_{(i,j)∈M} w(f̂_j)·S_ij / Σ_k w(f̂_k)
        # -----------------------------------------------------------------------
        r_flaw_num = sum(
            gt_weights[j] * sim_matrix[i, j]
            for i, j in matched_pairs
        )
        r_flaw = r_flaw_num / w_sum

        # -----------------------------------------------------------------------
        # Constraint 5: Gating
        # -----------------------------------------------------------------------
        gate = 1.0 if r_flaw >= GAMMA else 0.0

    # -----------------------------------------------------------------------
    # Constraint 2: False Positive Penalty
    # P_FP = λ_FP · Σ_{i∉M_pred} max(1.0, c_i)
    # -----------------------------------------------------------------------
    p_fp = LAMBDA_FP * fp * max(1.0, confidence)

    # -----------------------------------------------------------------------
    # Constraint 3: Critical Flaw Bonus
    # B_crit = λ_bonus · Σ_{(i,j)∈M} 𝟙[w(f̂_j)=1.0] · S_ij
    # -----------------------------------------------------------------------
    b_crit = 0.0
    if ground_truth_flaws and w_base >= 1.0:
        for i, j in matched_pairs:
            if gt_weights[j] >= 1.0:
                b_crit += LAMBDA_BONUS * sim_matrix[i, j]

    # -----------------------------------------------------------------------
    # Constraint 4: Severity-Weighted Recommendation Accuracy
    # d_base = |V(y_rec) - V(ŷ_rec)| / 3
    # A_rec = max(0, 1 - (d_base · e^{κ·w_max·𝟙[V(y)>V(ŷ)]}))
    # -----------------------------------------------------------------------
    v_pred = REC_ORDINAL.get(predicted_recommendation, 0)
    v_gt = REC_ORDINAL.get(ground_truth_recommendation, 0)
    d_base = abs(v_pred - v_gt) / 3.0

    if d_base == 0:
        a_rec = 1.0
    else:
        # Leniency: agent said accept/minor when should reject/major → exponential
        too_lenient = v_pred > v_gt
        if too_lenient:
            exponent = KAPPA * w_max * 1.0
            a_rec = max(0.0, 1.0 - d_base * math.exp(exponent))
        else:
            # Overly harsh: linear penalty only
            a_rec = max(0.0, 1.0 - d_base)

    # -----------------------------------------------------------------------
    # Unified formula
    # R_raw = Γ·(α·A_rec) + β·R_flaw + B_crit - P_FP
    # -----------------------------------------------------------------------
    r_raw = gate * (ALPHA * a_rec) + BETA * r_flaw + b_crit - p_fp

    # -----------------------------------------------------------------------
    # Normalization
    # R_max_possible = α + β + λ_bonus · (# critical flaws)
    # -----------------------------------------------------------------------
    n_critical = sum(1 for w in gt_weights if w >= 1.0) if ground_truth_flaws else 0
    r_max = ALPHA + BETA + LAMBDA_BONUS * max(n_critical, 1)
    r_total = max(0.0, min(1.0, r_raw / r_max))

    # -----------------------------------------------------------------------
    # Human-readable diagnostic
    # -----------------------------------------------------------------------
    details = (
        f"TP={tp} FP={fp} FN={fn} | "
        f"R_flaw={r_flaw:.3f} gate={gate} A_rec={a_rec:.3f} | "
        f"B_crit={b_crit:.3f} P_FP={p_fp:.3f} | "
        f"R_raw={r_raw:.3f} R_max={r_max:.3f} R_total={r_total:.3f}"
    )

    return RewardBreakdown(
        r_flaw=r_flaw,
        a_rec=a_rec,
        b_crit=b_crit,
        p_fp=p_fp,
        gate=gate,
        r_raw=r_raw,
        r_total=r_total,
        tp=tp,
        fp=fp,
        fn=fn,
        details=details,
    )
