from __future__ import annotations

import __future__
import ast
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
ROOT = Path(__file__).parents[1]


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.device = "cpu"

    def dim(self):
        return self.values.ndim

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.values, axis))

    def to(self, device):
        return self

    def t(self):
        return FakeTensor(self.values.T)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def item(self):
        return self.values.item()

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return (FakeTensor(row) for row in self.values)

    def __getitem__(self, item):
        return FakeTensor(self.values[item])


class FakeFunctional:
    @staticmethod
    def normalize(tensor, dim):
        values = tensor.values.astype(float)
        norms = np.linalg.norm(values, axis=dim, keepdims=True)
        norms[norms == 0] = 1.0
        return FakeTensor(values / norms)


class FakeTorch:
    class nn:
        functional = FakeFunctional()

    @staticmethod
    def tensor(values):
        return FakeTensor(values)

    @staticmethod
    def mm(left, right):
        return FakeTensor(np.matmul(left.values, right.values))

    @staticmethod
    def topk(tensor, k):
        indices = np.argsort(tensor.values)[::-1][:k]
        return FakeTensor(tensor.values[indices]), FakeTensor(indices)


def evaluate_evidence_support(score, threshold):
    passed = score is not None and score >= threshold
    return {
        "passed": passed,
        "status": "predicted" if passed else "insufficient_evidence",
        "support_score": score,
        "threshold": threshold,
        "reason": None if passed else "below_threshold",
        "message": "supported" if passed else "Insufficient evidence",
    }


def load_class_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    test_class = ast.ClassDef(
        name="ClassUnderTest",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
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
    return namespace["ClassUnderTest"]


class RecordingPubMed:
    def __init__(self):
        self.queries = []

    def search_by_metabolites(self, metabolites, **kwargs):
        self.queries.append(list(metabolites))
        return []


def test_mode1_queries_pubmed_with_five_ranked_metabolites():
    analyzer_class = load_class_method(
        ROOT / "Spec2Function" / "single_analysis.py",
        "SingleSpectrumAnalyzer",
        "single_inference",
        {
            "INSUFFICIENT_EVIDENCE_STATUS": "insufficient_evidence",
            "np": np,
            "torch": FakeTorch,
        },
    )
    analyzer = analyzer_class()
    analyzer.config = type("Config", (), {"min_similarity": 0.65})()
    analyzer._idx = {"e": {"spectrum": np.array([1.0, 0.0])}, "r": {}}
    analyzer._h = lambda *args: "spectrum"
    analyzer._sync = lambda: None
    analyzer.pubmed = RecordingPubMed()

    fragments = [
        {
            "accession": f"HMDB{i:07d}",
            "molecule_name": f"Metabolite {i}",
            "similarity": 0.90 - i / 100,
        }
        for i in range(6)
    ]
    analyzer.retrieve_similar_fragments_with_evidence = lambda *args, **kwargs: {
        "retrieved_fragments": fragments,
        "evidence_gate": evaluate_evidence_support(0.90, 0.65),
    }

    analyzer.single_inference([50.0], [1.0], precursor_mz=100.0)

    assert analyzer.pubmed.queries == [
        [f"Metabolite {i}" for i in range(5)]
    ]


def test_voting_retrieval_counts_one_vote_per_metabolite_and_spectrum():
    analyzer_class = load_class_method(
        ROOT / "Spec2Function" / "set_analysis.py",
        "MetaboliteSetAnalyzer",
        "retrieve_function_texts_by_voting_with_evidence",
        {
            "evaluate_evidence_support": evaluate_evidence_support,
            "torch": FakeTorch,
        },
    )
    analyzer = analyzer_class()
    analyzer.config = type("Config", (), {"min_similarity": 0.65})()
    analyzer.femdb_embeddings = FakeTensor(
        [
            [1.00, 0.00],
            [0.90, 0.10],
            [0.00, 1.00],
            [0.95, 0.05],
        ]
    )
    analyzer.femdb_fragments = [
        {"molecule_name": "A", "text": "A text 1"},
        {"molecule_name": "B", "text": "B text"},
        {"molecule_name": "C", "text": "C text"},
        {"molecule_name": "A", "text": "A text 2"},
    ]

    result = analyzer.retrieve_function_texts_by_voting_with_evidence(
        FakeTensor([[1.0, 0.0], [0.98, 0.02]]),
        top_k=2,
    )

    assert result["evidence_gate"]["passed"] is True
    assert [hit["name"] for hit in result["retrieved_texts"]] == ["A", "B"]
    assert [hit["vote_count"] for hit in result["retrieved_texts"]] == [2, 2]


def function_default(function, parameter):
    positional = list(function.args.posonlyargs) + list(function.args.args)
    offset = len(positional) - len(function.args.defaults)
    for index, argument in enumerate(positional):
        if argument.arg == parameter:
            return ast.literal_eval(function.args.defaults[index - offset])
    for argument, default in zip(
        function.args.kwonlyargs, function.args.kw_defaults
    ):
        if argument.arg == parameter:
            return ast.literal_eval(default)
    raise AssertionError(f"Parameter {parameter!r} not found")


def find_function(path, function_name, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = tree.body
    if class_name:
        nodes = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    return next(
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


@pytest.mark.parametrize(
    ("path", "class_name"),
    [
        (ROOT / "Spec2Function" / "set_analysis.py", "MetaboliteSetAnalyzer"),
        (ROOT / "Spec2Function" / "workflow.py", "MS2BioTextWorkflow"),
        (ROOT / "Spec2Function" / "workflow.py", None),
    ],
)
def test_mode2_public_api_defaults_to_centroid_top_five(path, class_name):
    function_name = (
        "run_semi_supervised_analysis"
        if class_name == "MetaboliteSetAnalyzer"
        else "run_set"
    )
    function = find_function(path, function_name, class_name)

    assert function_default(function, "retrieval_method") == "centroid"
    assert function_default(function, "top_k") == 5


@pytest.mark.parametrize(
    ("class_name", "called_method"),
    [
        ("MS2BioTextWorkflow", "run_semi_supervised_analysis"),
        (None, "run_set"),
    ],
)
def test_mode2_public_api_forwards_retrieval_options(class_name, called_method):
    function = find_function(
        ROOT / "Spec2Function" / "workflow.py",
        "run_set",
        class_name,
    )
    matching_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == called_method
    ]

    assert matching_calls
    keywords = {keyword.arg: keyword.value for keyword in matching_calls[0].keywords}
    assert isinstance(keywords["retrieval_method"], ast.Name)
    assert keywords["retrieval_method"].id == "retrieval_method"
    assert isinstance(keywords["top_k"], ast.Name)
    assert keywords["top_k"].id == "top_k"


@pytest.mark.parametrize(
    ("retrieval_method", "top_k", "message"),
    [
        ("nearest", 5, "retrieval_method"),
        ("centroid", 0, "top_k"),
        ("voting", True, "top_k"),
    ],
)
def test_mode2_rejects_invalid_retrieval_options(
    retrieval_method, top_k, message
):
    analyzer_class = load_class_method(
        ROOT / "Spec2Function" / "set_analysis.py",
        "MetaboliteSetAnalyzer",
        "run_semi_supervised_analysis",
        {},
    )

    with pytest.raises(ValueError, match=message):
        analyzer_class().run_semi_supervised_analysis(
            None,
            retrieval_method=retrieval_method,
            top_k=top_k,
        )
