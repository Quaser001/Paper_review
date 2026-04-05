"""
server/app.py — FastAPI application entry point for peer_review_env.

Uses create_app() factory pattern to guarantee thread-safe multi-tenant
WebSocket session isolation. Adds /ui endpoint for live dashboard and
/api/review-paper for user-uploaded PDF review.
"""
import io
import json
import logging
import os
import re
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_app

try:
    from ..models import PeerReviewAction, PeerReviewObservation
    from .peer_review_environment import PeerReviewEnvironment
except ImportError:
    from models import PeerReviewAction, PeerReviewObservation
    from server.peer_review_environment import PeerReviewEnvironment

logger = logging.getLogger(__name__)

# CRITICAL: pass the class, NOT an instance, so each WebSocket session
# gets its own isolated PeerReviewEnvironment state machine.
app = create_app(
    PeerReviewEnvironment,
    PeerReviewAction,
    PeerReviewObservation,
    env_name="peer_review_env",
)

# ---------------------------------------------------------------------------
# /ui — embedded dashboard (single-file HTML, no separate server needed)
# ---------------------------------------------------------------------------

_UI_HTML = None

def _load_ui_html() -> str:
    global _UI_HTML
    # Always reload in dev for fast iteration
    ui_path = Path(__file__).parent.parent / "ui" / "dashboard.html"
    if ui_path.exists():
        _UI_HTML = ui_path.read_text(encoding="utf-8")
    else:
        _UI_HTML = "<html><body><h1>Dashboard not found</h1></body></html>"
    return _UI_HTML


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def dashboard_ui():
    """Serve the live RL dashboard."""
    return HTMLResponse(content=_load_ui_html())


# ---------------------------------------------------------------------------
# /api/papers — expose paper metadata for dashboard
# ---------------------------------------------------------------------------

@app.get("/api/papers")
async def get_papers():
    """Return paper catalog for dashboard display."""
    data_path = Path(__file__).parent.parent / "data" / "papers.json"
    with open(data_path, "r") as f:
        papers = json.load(f)
    return [
        {
            "paper_id": p["paper_id"],
            "title": p["title"],
            "flaw_type": p["flaw_type"],
            "ground_truth_review": p["ground_truth_review"],
            "num_flaws": len(p["correct_flaws_list"]),
        }
        for p in papers
    ]


# ---------------------------------------------------------------------------
# /api/review-paper — Upload PDF → Extract text → LLM Review
# ---------------------------------------------------------------------------

REVIEW_SYSTEM_PROMPT = """You are an expert scientific peer reviewer with deep expertise in
machine learning, statistics, and research methodology.

You will be given a scientific paper (extracted text from a PDF). Your task is to produce a
thorough, structured peer review.

You MUST respond with valid JSON ONLY — no preamble, no markdown, no explanation outside the JSON.

JSON schema:
{
    "recommendation": "<accept|minor_revision|major_revision|reject>",
    "identified_flaws": ["<EXACT flaw name from the list below if found>", ...],
    "confidence": <float 0.0-1.0>,
    ...
}

Common flaws to check (YOU MUST USE THESE EXACT STRINGS in identified_flaws array):
- "p_hacking" (selective reporting, post-hoc threshold adjustments)
- "no_cross_validation" (evaluation on training data)
- "overclaimed_results" (abstract claims far exceed evidence)
- "fabricated_citation" (unverifiable or hallucinated citations)
- "impossible_statistics" (GRIM test failures, implausible p-values)

Be precise. Empty identified_flaws list if the paper is methodologically sound."""


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyPDF2."""
    import PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def _extract_sections(full_text: str) -> dict:
    """Attempt to split paper text into sections using common headers."""
    sections = {
        "title": "",
        "abstract": "",
        "methodology": "",
        "results": "",
        "conclusions": "",
        "full_text": full_text[:8000],  # Limit for LLM context
    }

    lines = full_text.split("\n")
    # Title is usually the first substantial line
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 10 and not line.lower().startswith(("arxiv", "http", "doi", "preprint")):
            sections["title"] = line
            break

    # Try to find abstract
    abstract_patterns = [
        r"(?i)abstract[:\s]*\n(.*?)(?=\n\s*(?:1[\.\s]|introduction|keywords))",
        r"(?i)abstract[:\s]*(.*?)(?=\n\s*(?:1[\.\s]|introduction|keywords))",
    ]
    for pat in abstract_patterns:
        m = re.search(pat, full_text, re.DOTALL)
        if m:
            sections["abstract"] = m.group(1).strip()[:2000]
            break

    # Section extraction via common headers
    section_map = {
        "methodology": [r"(?i)(?:method(?:ology|s)?|approach|model|framework)"],
        "results": [r"(?i)(?:results?|experiments?|evaluation|findings)"],
        "conclusions": [r"(?i)(?:conclusion|discussion|summary|future work)"],
    }
    for key, patterns in section_map.items():
        for pat in patterns:
            m = re.search(
                pat + r"[:\s]*\n(.*?)(?=\n\s*(?:\d+[\.\s]|references|bibliography|acknowledgment))",
                full_text, re.DOTALL
            )
            if m:
                sections[key] = m.group(1).strip()[:2000]
                break

    return sections


def _call_llm(paper_text: str) -> dict:
    """Call LLM to review the paper text using ultra-stable Gemini 2.5 backend."""
    import requests
    
    # Use user provided Gemini key for the unbreakable UI demo
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDT9SOFYH-mGxk6SyajscIf5crZXKbRDew")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    # Truncate to ~30,000 chars for context window (Gemini handles 1M+ but let's keep it safe)
    truncated = paper_text[:30000]

    system_instruction = {"parts": [{"text": REVIEW_SYSTEM_PROMPT}]}
    contents = [{"role": "user", "parts": [{"text": f"Please review this paper:\n\n{truncated}"}]}]
    
    # Enforce JSON output mode
    generation_config = {"response_mime_type": "application/json"}

    try:
        payload = {
            "system_instruction": system_instruction,
            "contents": contents,
            "generationConfig": generation_config
        }
        
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        
        data = r.json()
        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Parse Gemini's JSON
        return json.loads(raw)
        
    except json.JSONDecodeError:
        return {
            "recommendation": "major_revision",
            "identified_flaws": [],
            "confidence": 0.1,
            "reasoning": f"LLM returned invalid JSON. Raw response: {raw[:500]}",
            "summary": "Parse error during review.",
            "strengths": [],
            "weaknesses": ["Could not parse LLM output."]
        }
    except Exception as exc:
        return {
            "recommendation": "major_revision",
            "identified_flaws": [],
            "confidence": 0.0,
            "reasoning": f"Gemini API error: {exc}",
            "summary": "API call failed.",
            "strengths": [],
            "weaknesses": [str(exc)]
        }


@app.post("/api/review-paper")
async def review_uploaded_paper(file: UploadFile = File(...)):
    """
    Upload a PDF research paper for AI peer review.

    Returns structured review with recommendation, flaws, strengths, weaknesses.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported.", "status": "error"}

    try:
        file_bytes = await file.read()

        if len(file_bytes) > 20 * 1024 * 1024:  # 20MB limit
            return {"error": "File too large. Maximum size is 20MB.", "status": "error"}

        # Extract text
        raw_text = _extract_pdf_text(file_bytes)
        if not raw_text or len(raw_text.strip()) < 50:
            return {"error": "Could not extract text from PDF. It may be image-only.", "status": "error"}

        # Parse sections
        sections = _extract_sections(raw_text)

        # Call LLM for review
        review = _call_llm(raw_text)

        return {
            "status": "success",
            "filename": file.filename,
            "paper": {
                "title": sections["title"] or file.filename.replace(".pdf", ""),
                "abstract": sections["abstract"],
                "methodology": sections["methodology"],
                "results": sections["results"],
                "conclusions": sections["conclusions"],
                "text_length": len(raw_text),
                "pages": raw_text.count("\n\n") + 1,
            },
            "review": review,
        }

    except Exception as exc:
        logger.exception("Error processing uploaded paper")
        return {"error": str(exc), "status": "error"}


# ---------------------------------------------------------------------------
# /api/review-text — Review raw pasted text (no PDF needed)
# ---------------------------------------------------------------------------

@app.post("/api/review-text")
async def review_pasted_text(request: Request):
    """Review paper text pasted directly (no PDF upload needed)."""
    body = await request.json()
    text = body.get("text", "")
    if len(text.strip()) < 50:
        return {"error": "Text too short. Provide at least 50 characters.", "status": "error"}

    sections = _extract_sections(text)
    review = _call_llm(text)

    return {
        "status": "success",
        "filename": "pasted_text",
        "paper": {
            "title": sections["title"] or "Pasted Paper",
            "abstract": sections["abstract"],
            "methodology": sections["methodology"],
            "results": sections["results"],
            "conclusions": sections["conclusions"],
            "text_length": len(text),
        },
        "review": review,
    }


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
