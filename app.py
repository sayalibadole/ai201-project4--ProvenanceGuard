"""Provenance Guard — Flask backend.

Endpoints:
  * POST /submit  — classify text (Signal 1 + Signal 2 -> confidence -> label),
                    audit-log the decision, rate-limited against abuse.
  * POST /appeal  — contest a classification; flips status to under_review.
  * GET  /log     — return recent audit-log entries.
"""

from __future__ import annotations

from itertools import count

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from audit_log import find_submission, log_appeal, log_submission, read_log
from confidence import compute_confidence
from groq_detector import groq_classify
from labels import generate_label
from stylometric import stylometric_analyze

app = Flask(__name__)

# Rate limiting (per client IP). In-memory storage is fine for local/dev; a
# production deployment would point storage_uri at Redis so limits are shared
# across worker processes. See README for the chosen limits and rationale.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# Unique, monotonically increasing submission IDs. In-memory for now; this
# would move to persistent storage alongside the audit log in a real system.
_content_id_counter = count(1)


@app.post("/submit")
@limiter.limit("10 per minute;100 per day")
def submit():
    """Classify submitted text and return attribution, confidence, and label.

    Runs the detection pipeline (Signal 1 -> Signal 2 -> confidence scoring),
    generates a transparency label, and records the decision in the audit log.
    Rate-limited per IP (see README) — the limit is enforced before the
    (billable) Groq call.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    text = data.get("text")
    creator_id = data.get("creator_id")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required."}), 400
    if creator_id is None or (isinstance(creator_id, str) and not creator_id.strip()):
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    # Detection pipeline: Signal 1 -> Signal 2 -> Confidence scoring.
    signal_1 = groq_classify(text)
    signal_2 = stylometric_analyze(text)
    scoring = compute_confidence(signal_1["score"], signal_2["ai_likeness_score"])

    content_id = next(_content_id_counter)
    attribution = scoring["classification"]
    confidence = scoring["final_score"]
    label = generate_label(confidence)

    # Structured audit entry — both individual signal scores alongside the
    # combined confidence (timestamp + content_id added by the logger).
    log_submission(
        content_id=content_id,
        creator_id=creator_id,
        attribution=attribution,
        confidence=confidence,
        llm_score=signal_1["score"],
        stylometric_score=signal_2["ai_likeness_score"],
        status="classified",
    )

    return jsonify(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "attribution": attribution,
            "confidence": confidence,
            "label": label,
            "signals": {
                "groq": signal_1,
                "stylometric": signal_2,
            },
            "status": "classified",
        }
    ), 200


@app.get("/log")
def get_log():
    """Return the most recent audit-log entries as JSON.

    Optional `?limit=N` caps how many of the newest entries are returned.
    No auth here — this exists for documentation/grading visibility; a real
    system would protect it.
    """
    entries = read_log()
    entries.reverse()  # most recent first

    limit = request.args.get("limit", type=int)
    if limit is not None and limit >= 0:
        entries = entries[:limit]

    return jsonify({"entries": entries}), 200


@app.post("/appeal")
def appeal():
    """Contest a prior attribution. Flips status to 'under_review' and appends
    an appeal record to the audit log. Detection is NOT re-run.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    content_id = data.get("content_id")
    # Primary field is `creator_reasoning`; `reason` accepted as an alias.
    reasoning = data.get("creator_reasoning")
    if reasoning is None:
        reasoning = data.get("reason")

    if content_id is None or (isinstance(content_id, str) and not content_id.strip()):
        return jsonify({"error": "Field 'content_id' is required."}), 400
    if not isinstance(reasoning, str) or not reasoning.strip():
        return jsonify({"error": "Field 'creator_reasoning' is required."}), 400

    record = find_submission(content_id)
    if record is None:
        return jsonify({"error": f"No submission found for content_id {content_id!r}."}), 404

    # Append-only: preserve the original decision, record the contest.
    log_appeal(
        content_id=record["content_id"],
        original_classification=record["attribution"],
        confidence=record["confidence"],
        reason=reasoning,
    )

    return jsonify(
        {
            "message": "Appeal submitted successfully.",
            "content_id": record["content_id"],
            "status": "under_review",
        }
    ), 200


if __name__ == "__main__":
    app.run(debug=True)
