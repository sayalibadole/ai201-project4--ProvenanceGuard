"""Structured audit log (M3 — start simple).

Every submission writes one structured JSON object to an append-only
JSON Lines file (`audit_log.jsonl`). This is intentionally minimal; it gets
extended in M4 (both signal scores + final confidence) and M5 (appeals +
`GET /log` retrieval). Persistence may move to SQLite later, but the entry
shape stays the same.

Entry shape:
    {
      "content_id": int,
      "creator_id": str,
      "timestamp": ISO-8601 UTC with millisecond precision + 'Z',
      "attribution": "AI" | "Human" | "Uncertain",  # combined classification
      "confidence": float,            # combined final_score (0.65/0.35 weighted)
      "llm_score": float,             # Detection Signal 1 (Groq) score
      "stylometric_score": float,     # Detection Signal 2 (stylometric) score
      "status": str
    }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "audit_log.jsonl")


def _utc_now_iso() -> str:
    """Return the current UTC time as 'YYYY-MM-DDTHH:MM:SS.mmmZ'."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def log_submission(
    content_id: int,
    creator_id: str,
    attribution: str,
    confidence: float,
    llm_score: float,
    stylometric_score: float,
    status: str = "classified",
) -> dict:
    """Append one structured submission entry to the audit log.

    Captures both individual signal scores (`llm_score`, `stylometric_score`)
    alongside the combined `confidence`. Returns the entry that was written
    (useful for testing / response echoes).
    """
    entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": _utc_now_iso(),
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "stylometric_score": stylometric_score,
        "status": status,
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def read_log() -> list[dict]:
    """Read all audit-log entries (oldest first). Empty list if none yet."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def find_submission(content_id) -> dict | None:
    """Return the latest *submission* entry for `content_id`, or None.

    Appeal entries are skipped so this always resolves the original
    classification record. Comparison is type-tolerant (int vs str).
    """
    target = str(content_id)
    match = None
    for entry in read_log():
        if entry.get("type") == "appeal":
            continue
        if str(entry.get("content_id")) == target:
            match = entry  # keep the most recent submission for this id
    return match


def log_appeal(
    content_id,
    original_classification: str,
    confidence: float,
    reason: str,
) -> dict:
    """Append an appeal entry to the audit log.

    Append-only: the original submission entry is never mutated; the content's
    current status becomes the latest entry for that id (this appeal, with
    status 'under_review'). Detection is NOT re-run.
    """
    entry = {
        "type": "appeal",
        "content_id": content_id,
        "original_classification": original_classification,
        "confidence": confidence,
        "appeal_reasoning": reason,
        "status": "under_review",
        "timestamp": _utc_now_iso(),
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry
