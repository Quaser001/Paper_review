---
title: Peer Review RL Environment
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# peer_review_env

> **OpenEnv-compliant RL environment for automated scientific peer review**
> Built for Meta PyTorch OpenEnv Hackathon 2026

---

## Why This Exists

**~1 in 7 published papers contain statistical errors.** At NeurIPS 2025, over 100 fabricated citations were found across 51 accepted papers. Human peer review doesn't scale — there are ~5M papers published annually and reviewer fatigue is real.

`peer_review_env` trains LLM agents to perform rigorous, verifiable peer review. An agent receives a manuscript (abstract, methodology, results, conclusions, citations) and must:

1. **Identify specific methodological flaws** — not generic complaints, but precise failures like "Figure 3 reports mean=3.73 with N=10 which fails the GRIM test"
2. **Assign the correct editorial recommendation** (`accept / minor_revision / major_revision / reject`)
3. **Avoid hallucinating flaws** that don't exist — the reward function penalizes false positives

**This is uniquely suited for RL** because:
- Ground truth is verifiable (flaws are planted with mathematical certainty)
- The action space is structured JSON, not free text
- Partial credit through semantic matching rewards incremental improvement
- The gating function prevents reward hacking through shortcut strategies

---

## Flaw Taxonomy & Severity Weights

Each flaw type has a severity weight `w ∈ [0, 1]` that determines its contribution to the reward:

| Flaw Type | Severity `w` | Detection Method | Example |
|---|---|---|---|
| Formatting/verbose | 0.1 | Surface | Inconsistent font sizes |
| Scope exceedance | 0.3 | Semantic | Conclusions beyond data |
| Over-interpretation | 0.5 | Semantic | Causal claims from correlation |
| Methodological deviation | 0.6 | Logical | Using wrong statistical test |
| No cross-validation | 0.7 | Structural | Train=test environment |
| Overclaimed results | 0.5–0.8 | Comparative | Abstract vs. results mismatch |
| **P-hacking / seed selection** | **0.8** | **Statistical** | **"Best of 5 seeds" reported** |
| **Data fabrication** | **1.0** | **Verification** | **Fake DOIs, future dates** |
| **Impossible statistics (GRIM)** | **1.0** | **Mathematical** | **Mean of 3.73 with N=10** |

---

## Reward Function — Full Mathematical Specification

The reward function is designed to be **verifiable, anti-hackable, and granular**.

### Total Reward

```
R_total = max(0, min(1, R_raw / R_max_possible))
```

All rewards are normalized to `[0, 1]`.

### Raw Reward Decomposition

```
R_raw = Γ · (α · A_rec) + β · R_flaw + B_crit − P_FP
```

where:
- `α = 0.4` — recommendation accuracy weight
- `β = 0.6` — flaw detection recall weight
- `Γ` — gating function (anti-shortcut mechanism)

### Component 1: Flaw Detection Recall (`R_flaw`)

Flaws are matched using **SBERT semantic similarity** (`all-MiniLM-L6-v2`) with **Hungarian algorithm** optimal assignment:

```
For each (predicted_flaw, ground_truth_flaw) pair:
    cosine_sim = SBERT(predicted_flaw, ground_truth_flaw)
    match = 1 if cosine_sim ≥ τ (0.75)

R_flaw = Σ(w_i · matched_i) / Σ(w_i)    (severity-weighted recall)
```

This gives partial credit: finding 2 of 3 flaws earns proportional reward weighted by severity. Critical flaws (w=1.0) contribute more to recall than minor ones.

### Component 2: Gating Function (`Γ`)

```
Γ = 1 if R_flaw ≥ γ (0.4), else 0
```

**Why this matters:** Without gating, an agent could score points by always guessing "reject" without identifying any flaws. The gate ensures that the recommendation accuracy (`A_rec`) only counts if the agent has actually done the analytical work of finding flaws.

### Component 3: Recommendation Accuracy (`A_rec`)

```
ordinal_distance = |ord(predicted) - ord(ground_truth)|
    where: reject=0, major_revision=1, minor_revision=2, accept=3

Leniency penalty: worse if agent is too lenient on fraudulent papers
    penalty_multiplier = 1.0 + (w_max^κ) if predicted is more lenient
    κ = 1.5 (severity exponent)

A_rec = 1.0 - (ordinal_distance / 3) × leniency_modifier
```

### Component 4: Critical Flaw Bonus (`B_crit`)

```
B_crit = λ_bonus × (number of w=1.0 flaws correctly detected)
    λ_bonus = 0.3
```

Extra reward for catching the most dangerous flaws (fabricated data, impossible statistics).

### Component 5: False Positive Penalty (`P_FP`)

```
P_FP = λ_FP × (number of hallucinated flaws)
    λ_FP = 0.15 per false positive
```

Prevents "shotgun" strategies where the agent lists every possible flaw hoping some match.

### The GRIM Test

The **Granularity-Related Inconsistency of Means** (GRIM) test checks whether a reported mean is arithmetically possible given the sample size. For integer-valued data:

```
A mean of X with sample size N is valid iff:
    (X × N) mod 1 ≈ 0    (within floating-point tolerance)
```

Example: If a paper reports "mean accuracy = 73.3% across N=3 seeds", then `73.3 × 3 = 219.9`, which is not an integer. This is **mathematically impossible** and indicates data fabrication or reporting error.

Our environment embeds GRIM test failures in "hard" difficulty papers, requiring agents to perform this arithmetic verification.

---

## Tasks

| Task | Flaw Types | Papers | Expected Baseline | Expected Trained |
|---|---|---|---|---|
| **easy** | Correct papers (no flaws) | 5 | ~0.6 | ~0.9 |
| **medium** | p_hacking, no_cross_validation, overclaimed | 5 | ~0.2 | ~0.65 |
| **hard** | fabricated_citation, impossible_statistics | 5 | ~0.05 | ~0.45 |

---

## Action Space

```json
{
  "recommendation": "reject",
  "identified_flaws": [
    "P-hacking: methodology explicitly states iterative seed selection until p < 0.05",
    "GRIM test failure: mean=73.3% with N=3 is arithmetically impossible"
  ],
  "confidence": 0.9,
  "reasoning": "The methodology section states threshold adjustment post-hoc..."
}
```

## Observation Space

```json
{
  "paper": {
    "paper_id": "RL_STUB_002",
    "title": "...",
    "abstract": "...",
    "methodology": "...",
    "results": "...",
    "conclusions": "...",
    "citations": ["..."]
  },
  "feedback": "New paper loaded. Submit your peer review.",
  "task_name": "medium",
  "done": false
}
```

---

## Setup

```bash
# Install
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference baseline
export HF_TOKEN=your_token_here
python inference.py

# Validate
openenv validate --url http://localhost:7860

# Run graders
python graders.py --task easy --episodes 5
python graders.py --task medium --episodes 5
python graders.py --task hard --episodes 5
```

---

## Docker

```bash
docker build -t peer-review-env .
docker run -p 7860:7860 -e HF_TOKEN=your_token peer-review-env
```

---

## Curriculum Learning

The environment tracks agent rolling score over 5 episodes.
If score > 0.75 for 3 consecutive episodes twice in a row, `curriculum_level` increases:
- Level 1.0–1.5 → easy tasks
- Level 1.5–2.5 → easy + medium
- Level 2.5+ → medium + hard

Current level exposed in `state()` → `curriculum_level`.

---

## Infrastructure Constraints

- Runtime: < 20 minutes (single-turn episodes, fast inference)
- Memory: < 8GB (SBERT model: ~90MB, pre-baked in Docker image)
- CPU: 2 vCPUs sufficient (no GPU required)
- Python: 3.11+

---

## File Structure

```
/
├── inference.py          # Baseline LLM inference (emits [START][STEP][END])
├── graders.py            # Task graders (outputs JSON with "mean" key)
├── Dockerfile            # Docker build (port 7860, SBERT pre-baked)
├── openenv.yaml          # OpenEnv manifest (spec v1)
├── requirements.txt      # Python dependencies
├── models.py             # Pydantic Action/Observation/State contracts
├── reward.py             # SBERT + Hungarian matching reward function
├── client.py             # Typed async EnvClient
├── README.md             # This file
├── data/
│   └── papers.json       # 15 synthetic papers across all flaw types
├── ui/
│   └── dashboard.html    # Embedded RL dashboard (served at /ui)
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI create_app + /ui + /api endpoints
    └── peer_review_environment.py  # Core Environment state machine
```

---

## License

MIT
