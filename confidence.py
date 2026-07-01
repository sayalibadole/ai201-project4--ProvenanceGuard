"""Confidence Scoring Engine (M4).

Combines Signal 1 (Groq LLM) and Signal 2 (stylometric) into a single
probability and classification. Pure function, no external dependencies.

    compute_confidence(groq_score: float, stylometric_score: float) -> dict

Mandatory weighted formula:
    final_score = (0.65 * groq_score) + (0.35 * stylometric_score)

Threshold mapping (must match the planning doc exactly):
    0.00 – 0.39  -> "Human"
    0.40 – 0.69  -> "Uncertain"
    0.70 – 1.00  -> "AI"
"""

from __future__ import annotations

GROQ_WEIGHT = 0.65
STYLOMETRIC_WEIGHT = 0.35

HUMAN_UPPER = 0.40   # final_score < 0.40            -> Human
UNCERTAIN_UPPER = 0.70  # 0.40 <= final_score < 0.70 -> Uncertain; >= 0.70 -> AI


def compute_confidence(groq_score: float, stylometric_score: float) -> dict:
    """Return {"final_score": float, "classification": "AI"|"Human"|"Uncertain"}."""
    final_score = (GROQ_WEIGHT * groq_score) + (STYLOMETRIC_WEIGHT * stylometric_score)

    if final_score < HUMAN_UPPER:
        classification = "Human"
    elif final_score < UNCERTAIN_UPPER:
        classification = "Uncertain"
    else:
        classification = "AI"

    return {"final_score": round(final_score, 4), "classification": classification}


if __name__ == "__main__":
    cases = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.8, 0.2), (0.2, 0.8)]
    for g, s in cases:
        print(f"groq={g}, stylo={s} -> {compute_confidence(g, s)}")
