from __future__ import annotations

import __future__
import ast
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
sklearn_cluster = pytest.importorskip("sklearn.cluster")
sklearn_decomposition = pytest.importorskip("sklearn.decomposition")
sklearn_manifold = pytest.importorskip("sklearn.manifold")
sklearn_metrics = pytest.importorskip("sklearn.metrics")

ROOT = Path(__file__).parents[1]


def _load_gate():
    path = ROOT / "Spec2Function" / "evidence_gate.py"
    spec = importlib.util.spec_from_file_location("legacy_manager_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_manager_method(gate):
    path = ROOT / "Spec2Function" / "model_manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    manager_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Spec2FunctionManager"
    )
    function = next(
        node
        for node in manager_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_semi_supervised_analysis"
    )
    function.decorator_list = []
    test_class = ast.ClassDef(
        name="ManagerUnderTest",
        bases=[],
        keywords=[],
        body=[function],
        decorator_list=[],
    )
    namespace = {
        "Counter": Counter,
        "DEFAULT_EVIDENCE_THRESHOLD": gate.DEFAULT_EVIDENCE_THRESHOLD,
        "INSUFFICIENT_EVIDENCE_STATUS": gate.INSUFFICIENT_EVIDENCE_STATUS,
        "KMeans": sklearn_cluster.KMeans,
        "PCA": sklearn_decomposition.PCA,
        "PREDICTED_STATUS": gate.PREDICTED_STATUS,
        "TSNE": sklearn_manifold.TSNE,
        "defaultdict": defaultdict,
        "evaluate_evidence_support": gate.evaluate_evidence_support,
        "np": np,
        "pd": pd,
        "re": re,
        "silhouette_score": sklearn_metrics.silhouette_score,
        "torch": SimpleNamespace(tensor=lambda values: np.asarray(values)),
    }
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    exec(
        compile(
            module,
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace["ManagerUnderTest"]


def _load_set_analyzer_method(gate):
    path = ROOT / "Spec2Function" / "set_analysis.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    analyzer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MetaboliteSetAnalyzer"
    )
    function = next(
        node
        for node in analyzer_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_semi_supervised_analysis"
    )
    function.decorator_list = []
    test_class = ast.ClassDef(
        name="SetAnalyzerUnderTest",
        bases=[],
        keywords=[],
        body=[function],
        decorator_list=[],
    )
    namespace = {
        "DEFAULT_EVIDENCE_THRESHOLD": gate.DEFAULT_EVIDENCE_THRESHOLD,
        "INSUFFICIENT_EVIDENCE_STATUS": gate.INSUFFICIENT_EVIDENCE_STATUS,
        "KMeans": sklearn_cluster.KMeans,
        "PCA": sklearn_decomposition.PCA,
        "PREDICTED_STATUS": gate.PREDICTED_STATUS,
        "TSNE": sklearn_manifold.TSNE,
        "evaluate_evidence_support": gate.evaluate_evidence_support,
        "np": np,
        "pd": pd,
        "re": re,
        "silhouette_score": sklearn_metrics.silhouette_score,
        "torch": SimpleNamespace(tensor=lambda values: np.asarray(values)),
    }
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    exec(
        compile(
            module,
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace["SetAnalyzerUnderTest"]


class _Batch:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _NoCallService:
    def __init__(self):
        self.calls = 0

    def __getattr__(self, name):
        def fail_if_called(*args, **kwargs):
            self.calls += 1
            raise AssertionError(f"{name} must not be called")

        return fail_if_called


def test_legacy_mode2_refuses_all_clusters_without_any_llm_call():
    gate = _load_gate()
    manager_class = _load_manager_method(gate)
    manager = manager_class()
    manager.config = SimpleNamespace(min_similarity=0.65)
    manager.gpt = _NoCallService()
    manager.pubmed = _NoCallService()

    rng = np.random.default_rng(7)
    embeddings = np.vstack(
        [
            rng.normal(-3.0, 0.05, size=(10, 4)),
            rng.normal(3.0, 0.05, size=(10, 4)),
        ]
    )
    manager.encode_ms2_batch = lambda parsed: _Batch(embeddings)
    thresholds = []

    def no_supported_hits(centroid, top_k=5, min_similarity=None):
        thresholds.append(min_similarity)
        return []

    manager.retrieve_function_texts_from_centroid = no_supported_hits
    manager.retrieve_function_texts_by_voting = lambda *args, **kwargs: []

    data = pd.DataFrame(
        {
            "variable_id": [f"feature_{i}" for i in range(20)],
            "ms2_spectrum_string": [
                f"50:{i + 1}|75:{i + 2}|100:{i + 3}" for i in range(20)
            ],
            "logFC": np.linspace(-1.0, 1.0, 20),
            "precursor_mz": np.linspace(100.0, 119.0, 20),
        }
    )

    result = manager.run_semi_supervised_analysis(data)

    assert "error" not in result
    assert result["status"] == "insufficient_evidence"
    assert result["llm_called"] is False
    assert result["evidence_gate"]["threshold"] == 0.65
    assert result["evidence_gate"]["n_predicted_clusters"] == 0
    assert result["clusters"]
    assert all(c["status"] == "insufficient_evidence" for c in result["clusters"])
    assert manager.gpt.calls == 0
    assert manager.pubmed.calls == 0
    assert thresholds and all(value == 0.65 for value in thresholds)


def test_public_run_set_analyzer_refuses_before_any_llm_call():
    gate = _load_gate()
    analyzer_class = _load_set_analyzer_method(gate)
    analyzer = analyzer_class()
    analyzer.config = SimpleNamespace(min_similarity=0.65)
    analyzer.gpt = _NoCallService()
    analyzer.pubmed = _NoCallService()

    rng = np.random.default_rng(9)
    embeddings = np.vstack(
        [
            rng.normal(-2.0, 0.05, size=(10, 4)),
            rng.normal(2.0, 0.05, size=(10, 4)),
        ]
    )
    analyzer.encode_ms2_batch = lambda parsed: _Batch(embeddings)

    def unsupported_retrieval(centroid, top_k=10, min_similarity=None):
        return {
            "retrieved_texts": [],
            "evidence_gate": gate.evaluate_evidence_support(0.64, min_similarity),
        }

    analyzer.retrieve_function_texts_with_evidence = unsupported_retrieval
    data = pd.DataFrame(
        {
            "variable_id": [f"feature_{i}" for i in range(20)],
            "ms2_spectrum_string": [
                f"50:{i + 1}|75:{i + 2}|100:{i + 3}" for i in range(20)
            ],
            "logFC": np.linspace(-1.0, 1.0, 20),
            "precursor_mz": np.linspace(100.0, 119.0, 20),
        }
    )

    result = analyzer.run_semi_supervised_analysis(data)

    assert "error" not in result
    assert result["status"] == "insufficient_evidence"
    assert result["llm_called"] is False
    assert result["evidence_gate"]["threshold"] == 0.65
    assert all(c["support_score"] == 0.64 for c in result["clusters"])
    assert analyzer.gpt.calls == 0
    assert analyzer.pubmed.calls == 0
