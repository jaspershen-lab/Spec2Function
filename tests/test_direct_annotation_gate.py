from __future__ import annotations

import __future__
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]


def _load_gate():
    path = ROOT / "Spec2Function" / "evidence_gate.py"
    spec = importlib.util.spec_from_file_location("direct_annotation_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_generate_annotation(gate):
    path = ROOT / "Spec2Function" / "single_analysis.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    analyzer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SingleSpectrumAnalyzer"
    )
    function = next(
        node
        for node in analyzer.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_annotation"
    )
    function.decorator_list = []
    namespace = {
        "evaluate_evidence_support": gate.evaluate_evidence_support,
        "np": SimpleNamespace(isfinite=math.isfinite),
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace["generate_annotation"]


class _GPT:
    def __init__(self):
        self.calls = 0

    def single_annotation(self, **kwargs):
        self.calls += 1
        return "generated"


def test_direct_annotation_refuses_below_threshold_without_llm_call():
    gate = _load_gate()
    generate_annotation = _load_generate_annotation(gate)
    gpt = _GPT()
    analyzer = SimpleNamespace(
        config=SimpleNamespace(min_similarity=0.65),
        gpt=gpt,
    )

    refused = generate_annotation(
        analyzer,
        [{"similarity": 0.64, "text": "unsupported"}],
        threshold=0.65,
    )
    generated = generate_annotation(
        analyzer,
        [{"similarity": 0.66, "text": "supported"}],
        threshold=0.65,
    )

    assert refused.startswith("Insufficient evidence:")
    assert generated == "generated"
    assert gpt.calls == 1
