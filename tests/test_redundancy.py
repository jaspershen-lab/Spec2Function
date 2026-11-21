from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

ROOT = Path(__file__).parents[1]

GLUCOSE = 180.063388  # neutral monoisotopic mass
H, NA, K, NH4 = 1.007276, 22.989218, 38.963158, 18.033823


def _load_redundancy():
    path = ROOT / "Spec2Function" / "redundancy.py"
    spec = importlib.util.spec_from_file_location("redundancy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


red = _load_redundancy()


def test_adducts_and_isotopes_of_one_compound_are_linked():
    ions = red.COMMON_ION_TYPES["positive"]
    mh, mna = GLUCOSE + H, GLUCOSE + NA
    assert red.ion_relation(mh, mna, ions) is not None
    assert red.ion_relation(mh, GLUCOSE + K, ions) is not None
    assert red.ion_relation(mh, GLUCOSE + NH4, ions) is not None
    assert red.ion_relation(mh, 2 * GLUCOSE + H, ions) is not None
    assert red.ion_relation(mh, mh + 1.003355, ions) == "13C"
    assert red.ion_relation(mh, mh + 0.5, ions) is None


def test_grouping_requires_co_elution():
    mz = [GLUCOSE + H, GLUCOSE + NA, GLUCOSE + NA, 300.1]
    rt = [100.0, 103.0, 200.0, 101.0]
    labels, links = red.group_redundant_ions(mz, rt, rt_tolerance=5.0)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    assert labels[3] != labels[0]
    assert len(links) == 1


def test_negative_mode_uses_negative_ion_types():
    mz = [GLUCOSE - H, GLUCOSE + 34.969402]  # [M-H]-, [M+Cl]-
    rt = [50.0, 51.0]
    labels, _ = red.group_redundant_ions(mz, rt, ion_mode="negative")
    assert labels[0] == labels[1]
    labels, _ = red.group_redundant_ions(mz, rt, ion_mode="positive")
    assert labels[0] != labels[1]


def _frame(modules):
    return pd.DataFrame(
        {
            "variable_id": [
                f"mz{GLUCOSE + H:.4f}rt100.0",
                f"mz{GLUCOSE + NA:.4f}rt102.0",
                f"mz{GLUCOSE + K:.4f}rt101.0",
                "mz300.1000rt100.0",
            ],
            "cluster_id": modules,
        }
    )


def _spectra(tics):
    return [{"mz": [50.0], "intensity": [t]} for t in tics]


def test_largest_tic_ion_is_kept_within_each_module():
    df = _frame([0, 0, 0, 0])
    summary = red.mark_redundant_features(df, list(df.index), _spectra([10, 50, 20, 5]))
    assert summary["applied"] and summary["n_redundant"] == 2
    assert df.is_redundant.tolist() == [True, False, True, False]
    assert df.representative_id[0] == df.variable_id[1]


def test_ions_in_different_modules_are_not_removed():
    df = _frame([0, 1, 2, 0])
    summary = red.mark_redundant_features(df, list(df.index), _spectra([10, 50, 20, 5]))
    assert summary["n_redundant"] == 0
    assert not df.is_redundant.any()


def test_explicit_mz_rt_columns_and_minutes_are_supported():
    df = pd.DataFrame(
        {
            "variable_id": ["a", "b", "c"],
            "mz": [GLUCOSE + H, GLUCOSE + NA, 300.1],
            "rt": [1.50, 1.53, 1.50],  # minutes
            "cluster_id": [0, 0, 0],
        }
    )
    summary = red.mark_redundant_features(df, list(df.index), _spectra([1, 2, 3]))
    assert "minutes" in summary["rt_source"]
    assert df.is_redundant.tolist() == [True, False, False]


def test_missing_rt_skips_removal():
    df = pd.DataFrame({"variable_id": ["a", "b"], "precursor_mz": [181.07, 203.05], "cluster_id": [0, 0]})
    summary = red.mark_redundant_features(df, list(df.index), _spectra([1, 2]))
    assert summary["applied"] is False
    assert "skipped" in summary["message"]
    assert not df.is_redundant.any()


# ---------------------------------------------------------------------------
# Mode 2 end to end with a fake encoder (real package import)
# ---------------------------------------------------------------------------


def _mode2_data():
    """20 features in two well-separated groups; group A contains 3 ions of glucose."""
    rows = []
    for i in range(20):
        rows.append(
            {
                "variable_id": f"mz{150.0 + 7.3 * i:.4f}rt{60.0 + 40.0 * i:.1f}",
                "ms2_spectrum_string": f"50:{i + 1}|75:{i + 2}|100:{i + 3}",
                "logFC": 1.0 if i < 10 else -1.0,
                "precursor_mz": 150.0 + 7.3 * i,
            }
        )
    # three co-eluting ions of glucose in group A (rows 0-2), [M+Na]+ has the largest TIC
    rows[0].update(variable_id=f"mz{GLUCOSE + H:.4f}rt500.0", ms2_spectrum_string="50:1|75:1", logFC=1.0)
    rows[1].update(variable_id=f"mz{GLUCOSE + NA:.4f}rt501.0", ms2_spectrum_string="50:100|75:100", logFC=1.0)
    rows[2].update(variable_id=f"mz{GLUCOSE + K:.4f}rt502.0", ms2_spectrum_string="50:2|75:2", logFC=1.0)
    return pd.DataFrame(rows)


def _embeddings():
    rng = np.random.default_rng(3)
    return np.vstack([rng.normal(-3.0, 0.05, size=(10, 4)), rng.normal(3.0, 0.05, size=(10, 4))])


class _Batch:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def _import_real_set_analysis(monkeypatch):
    """Other test modules register stub Spec2Function.* modules; import the real ones."""
    import importlib
    import sys

    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    pytest.importorskip("transformers")
    for name in [m for m in sys.modules if m == "Spec2Function" or m.startswith("Spec2Function.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(ROOT))
    set_analysis = importlib.import_module("Spec2Function.set_analysis")
    gate = importlib.import_module("Spec2Function.evidence_gate")
    return set_analysis, gate.evaluate_evidence_support


def _run_set_analyzer(monkeypatch, remove_redundancy):
    set_analysis, evaluate_evidence_support = _import_real_set_analysis(monkeypatch)

    analyzer = object.__new__(set_analysis.MetaboliteSetAnalyzer)
    analyzer.config = SimpleNamespace(min_similarity=0.65)
    analyzer.gpt = None
    analyzer.pubmed = None
    analyzer.encode_ms2_batch = lambda parsed: _Batch(_embeddings())
    sizes = []

    def retrieval(centroid, top_k=5, min_similarity=None):
        sizes.append(centroid)
        return {"retrieved_texts": [], "evidence_gate": evaluate_evidence_support(0.9, min_similarity)}

    analyzer.retrieve_function_texts_with_evidence = retrieval
    result = analyzer.run_semi_supervised_analysis(_mode2_data(), remove_redundancy=remove_redundancy)
    return result


def test_mode2_default_keeps_all_features(monkeypatch):
    result = _run_set_analyzer(monkeypatch, remove_redundancy=False)
    assert "error" not in result
    assert result["redundancy_removal"]["enabled"] is False
    assert all(c["n_redundant_removed"] == 0 for c in result["clusters"])
    assert "is_redundant" not in result["plot_data"][0]


def test_mode2_remove_redundancy_drops_extra_ions_from_annotation(monkeypatch):
    result = _run_set_analyzer(monkeypatch, remove_redundancy=True)
    assert "error" not in result
    summary = result["redundancy_removal"]
    assert summary["applied"] is True
    assert summary["n_redundant"] == 2
    assert sum(c["n_redundant_removed"] for c in result["clusters"]) == 2
    group_a = next(c for c in result["clusters"] if c["n_redundant_removed"] == 2)
    assert group_a["n_features"] == 10
    assert group_a["known_count"] + group_a["unknown_count"] == 8
    # every feature stays in the plot, the kept ion of glucose is the one with the largest TIC
    points = {p["variable_id"]: p for p in result["plot_data"]}
    assert len(points) == 20
    kept = f"mz{GLUCOSE + NA:.4f}rt501.0"
    assert points[kept]["is_redundant"] is False
    assert points[f"mz{GLUCOSE + H:.4f}rt500.0"]["representative_id"] == kept
    assert points[f"mz{GLUCOSE + K:.4f}rt502.0"]["is_redundant"] is True
