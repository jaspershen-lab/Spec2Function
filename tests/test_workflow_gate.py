import importlib.util
import sys
import types
from pathlib import Path


PACKAGE_DIR = Path(__file__).parents[1] / "Spec2Function"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


package = types.ModuleType("Spec2Function")
package.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault("Spec2Function", package)

pandas = types.ModuleType("pandas")
pandas.DataFrame = object
pandas.read_csv = lambda path: None
sys.modules.setdefault("pandas", pandas)

single_analysis = types.ModuleType("Spec2Function.single_analysis")
single_analysis.SingleSpectrumAnalyzer = object
sys.modules.setdefault("Spec2Function.single_analysis", single_analysis)

set_analysis = types.ModuleType("Spec2Function.set_analysis")
set_analysis.MetaboliteSetAnalyzer = object
sys.modules.setdefault("Spec2Function.set_analysis", set_analysis)

evidence_gate = _load_module(
    "Spec2Function.evidence_gate", PACKAGE_DIR / "evidence_gate.py"
)
workflow_module = _load_module("Spec2Function.workflow", PACKAGE_DIR / "workflow.py")
MS2BioTextWorkflow = workflow_module.MS2BioTextWorkflow


PAYLOAD = {"precursor_mz": 100.0, "peaks": [[50.0, 1.0]]}


class FakeAnalyzer:
    def __init__(self, score):
        self.score = score
        self.gpt = object()
        self.annotation_calls = 0
        self.received_threshold = None

    def single_inference(
        self,
        mz,
        intensity,
        precursor_mz=None,
        top_k=10,
        min_similarity=None,
    ):
        threshold = 0.65 if min_similarity is None else min_similarity
        self.received_threshold = threshold
        gate = evidence_gate.evaluate_evidence_support(self.score, threshold)
        fragments = [{"similarity": self.score}] if gate["passed"] else []
        return {
            "status": gate["status"],
            "evidence_gate": gate,
            "support_threshold": threshold,
            "retrieved_fragments": fragments,
            "papers": [],
            "annotation": gate["message"] if not gate["passed"] else None,
            "llm_called": False,
        }

    def generate_annotation(self, **kwargs):
        self.annotation_calls += 1
        return "generated"


def test_run_single_refuses_without_calling_llm():
    analyzer = FakeAnalyzer(score=0.64)
    result = MS2BioTextWorkflow(analyzer=analyzer).run_single(PAYLOAD)

    assert result["status"] == "insufficient_evidence"
    assert result["llm_called"] is False
    assert analyzer.annotation_calls == 0


def test_run_single_propagates_override_and_calls_llm_after_gate_passes():
    analyzer = FakeAnalyzer(score=0.64)
    result = MS2BioTextWorkflow(analyzer=analyzer).run_single(
        PAYLOAD,
        min_similarity=0.60,
    )

    assert analyzer.received_threshold == 0.60
    assert result["status"] == "predicted"
    assert result["annotation"] == "generated"
    assert result["llm_called"] is True
    assert analyzer.annotation_calls == 1
