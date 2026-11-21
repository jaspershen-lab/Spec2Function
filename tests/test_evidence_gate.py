import importlib.util
import math
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "Spec2Function" / "evidence_gate.py"
SPEC = importlib.util.spec_from_file_location("spec2function_evidence_gate", MODULE_PATH)
evidence_gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evidence_gate)


def test_default_gate_boundary_and_metadata():
    threshold = evidence_gate.DEFAULT_EVIDENCE_THRESHOLD
    passed = evidence_gate.evaluate_evidence_support(threshold)
    refused = evidence_gate.evaluate_evidence_support(threshold - 1e-6)

    assert passed["status"] == evidence_gate.PREDICTED_STATUS
    assert passed["passed"] is True
    assert passed["threshold_overridden"] is False
    assert refused["status"] == evidence_gate.INSUFFICIENT_EVIDENCE_STATUS
    assert refused["reason"] == "maximum_femdb_similarity_below_threshold"


def test_gate_supports_explicit_override_and_rejects_invalid_thresholds():
    decision = evidence_gate.evaluate_evidence_support(0.61, threshold=0.60)

    assert decision["passed"] is True
    assert decision["threshold"] == 0.60
    assert decision["threshold_overridden"] is True
    assert evidence_gate.evaluate_evidence_support(math.nan)["passed"] is False

    with pytest.raises(ValueError):
        evidence_gate.evaluate_evidence_support(0.8, threshold=math.inf)
    with pytest.raises(ValueError):
        evidence_gate.evaluate_evidence_support(0.8, threshold=1.1)
