"""Structured decisions for the Spec2Function evidence-support gate."""

from __future__ import annotations

import math
from typing import Dict, Optional


DEFAULT_EVIDENCE_THRESHOLD = 0.65
PREDICTED_STATUS = "predicted"
INSUFFICIENT_EVIDENCE_STATUS = "insufficient_evidence"
SUPPORT_METRIC = "maximum_spectrum_to_femdb_cosine"


def evaluate_evidence_support(
    support_score: Optional[float],
    threshold: float = DEFAULT_EVIDENCE_THRESHOLD,
) -> Dict:
    """Return a JSON-serializable evidence-gate decision.

    The default threshold is the manuscript value (0.65). Callers may
    explicitly override it for exploratory analyses. The score measures
    retrieval support only and is not a correctness probability.
    """

    threshold = float(threshold)
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite cosine value between -1 and 1")

    score = None if support_score is None else float(support_score)
    if score is not None and not math.isfinite(score):
        score = None

    passed = score is not None and score >= threshold
    if passed:
        reason = None
        message = (
            f"Evidence-support gate passed: maximum spectrum-to-FemDB cosine "
            f"{score:.4f} is at least the configured threshold {threshold:.2f}. "
            "This gate does not guarantee annotation correctness."
        )
        status = PREDICTED_STATUS
    else:
        status = INSUFFICIENT_EVIDENCE_STATUS
        if score is None:
            reason = "no_femdb_candidates"
            message = (
                "Insufficient evidence: no valid FemDB support score was "
                "available. No functional annotation was generated."
            )
        else:
            reason = "maximum_femdb_similarity_below_threshold"
            message = (
                f"Insufficient evidence: maximum spectrum-to-FemDB cosine "
                f"{score:.4f} is below the configured {threshold:.2f} "
                "evidence-support threshold. No functional annotation was generated."
            )

    return {
        "status": status,
        "passed": passed,
        "support_metric": SUPPORT_METRIC,
        "support_score": score,
        "threshold": threshold,
        "default_threshold": DEFAULT_EVIDENCE_THRESHOLD,
        "threshold_overridden": threshold != DEFAULT_EVIDENCE_THRESHOLD,
        "reason": reason,
        "message": message,
        "interpretation": "evidence_support_only_not_correctness_guarantee",
    }
