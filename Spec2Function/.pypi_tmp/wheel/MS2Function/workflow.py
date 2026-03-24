from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import json

import pandas as pd

from .single_analysis import SingleSpectrumAnalyzer
from .set_analysis import MetaboliteSetAnalyzer

SpectrumInput = Union[str, Dict[str, Any]]
CSVInput = Union[str, Path, pd.DataFrame]


def parse_json_spectrum(json_input: SpectrumInput) -> Dict[str, Any]:
    if isinstance(json_input, (str, bytes, bytearray)):
        data = json.loads(json_input)
    elif isinstance(json_input, dict):
        data = json_input
    else:
        raise TypeError("json_input must be a JSON string or a dict.")

    peaks = data.get("peaks") or []
    if not peaks:
        raise ValueError("No peaks found in JSON input.")

    mz_values = []
    intensity_values = []
    for peak in peaks:
        if not isinstance(peak, (list, tuple)) or len(peak) < 2:
            continue
        mz_values.append(float(peak[0]))
        intensity_values.append(float(peak[1]))

    if not mz_values:
        raise ValueError("No valid m/z-intensity pairs found in JSON input.")

    precursor_mz = float(data.get("precursor_mz") or 0.0)
    return {
        "mz": mz_values,
        "intensity": intensity_values,
        "precursor_mz": precursor_mz,
    }


def load_set_dataframe(data: CSVInput) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(Path(data))


def filter_set_dataframe(
    df: pd.DataFrame,
    min_abs_logfc: float = 0.1,
    max_pvalue: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    filtered = df.copy()
    columns = list(filtered.columns)
    logfc_col = next((c for c in columns if "logfc" in c.lower()), None)
    pval_col = next(
        (c for c in columns if "pval" in c.lower() or "p.value" in c.lower()),
        None,
    )

    if logfc_col and min_abs_logfc and min_abs_logfc > 0:
        filtered = filtered[filtered[logfc_col].abs() >= min_abs_logfc]

    if pval_col and max_pvalue is not None and max_pvalue < 1:
        filtered = filtered[filtered[pval_col] <= max_pvalue]

    info = {
        "total": len(df),
        "kept": len(filtered),
        "logfc_col": logfc_col,
        "pval_col": pval_col,
        "min_abs_logfc": min_abs_logfc,
        "max_pvalue": max_pvalue,
    }
    return filtered, info


def _filter_papers(papers: Iterable[Dict[str, Any]], selected_pmids: Optional[Iterable[str]]) -> list:
    if not selected_pmids:
        return list(papers or [])
    selected = {str(pmid) for pmid in selected_pmids}
    return [p for p in (papers or []) if str(p.get("pmid")) in selected]


def _ensure_analyzer(
    analyzer: Optional[MetaboliteSetAnalyzer],
    project_root: Optional[Path],
    *,
    device=None,
    enable_gpt_pubmed: bool = True,
) -> MetaboliteSetAnalyzer:
    if analyzer is not None:
        return analyzer
    if project_root is None:
        raise ValueError("project_root is required when analyzer is not provided.")
    return MetaboliteSetAnalyzer.create_from_ms2function_root(
        project_root,
        device=device,
        enable_gpt_pubmed=enable_gpt_pubmed,
    )


@dataclass
class MS2BioTextWorkflow:
    analyzer: MetaboliteSetAnalyzer

    @classmethod
    def from_ms2function_root(
        cls,
        project_root: Path,
        *,
        device=None,
        enable_gpt_pubmed: bool = True,
    ) -> "MS2BioTextWorkflow":
        analyzer = MetaboliteSetAnalyzer.create_from_ms2function_root(
            project_root,
            device=device,
            enable_gpt_pubmed=enable_gpt_pubmed,
        )
        return cls(analyzer=analyzer)

    def run_single(
        self,
        json_input: SpectrumInput,
        *,
        top_k: int = 10,
        user_focus: Optional[str] = None,
        selected_pmids: Optional[Iterable[str]] = None,
        include_annotation: bool = True,
    ) -> Dict[str, Any]:
        spectrum = parse_json_spectrum(json_input)
        result = self.analyzer.single_inference(
            spectrum["mz"],
            spectrum["intensity"],
            precursor_mz=spectrum["precursor_mz"],
            top_k=top_k,
        )

        if include_annotation:
            papers = _filter_papers(result.get("papers", []), selected_pmids)
            annotation = self.analyzer.generate_annotation(
                retrieved_fragments=result.get("retrieved_fragments", []),
                papers=papers,
                user_focus=user_focus,
            )
            result["annotation"] = annotation
        return result

    def run_set(
        self,
        data: CSVInput,
        *,
        background_info: Optional[str] = None,
        min_abs_logfc: float = 0.1,
        max_pvalue: float = 0.05,
        min_features: int = 5,
    ) -> Dict[str, Any]:
        df = load_set_dataframe(data)
        filtered, info = filter_set_dataframe(df, min_abs_logfc, max_pvalue)

        if len(filtered) < min_features:
            return {
                "error": f"Too few features selected ({len(filtered)})",
                "filter": info,
            }

        result = self.analyzer.run_semi_supervised_analysis(
            filtered, background_info=background_info
        )
        result["filter"] = info
        return result


def run_single(
    json_input: SpectrumInput,
    *,
    project_root: Optional[Path] = None,
    analyzer: Optional[MetaboliteSetAnalyzer] = None,
    device=None,
    enable_gpt_pubmed: bool = True,
    top_k: int = 10,
    user_focus: Optional[str] = None,
    selected_pmids: Optional[Iterable[str]] = None,
    include_annotation: bool = True,
) -> Dict[str, Any]:
    analyzer = _ensure_analyzer(
        analyzer,
        project_root,
        device=device,
        enable_gpt_pubmed=enable_gpt_pubmed,
    )
    workflow = MS2BioTextWorkflow(analyzer=analyzer)
    return workflow.run_single(
        json_input,
        top_k=top_k,
        user_focus=user_focus,
        selected_pmids=selected_pmids,
        include_annotation=include_annotation,
    )


def run_set(
    data: CSVInput,
    *,
    project_root: Optional[Path] = None,
    analyzer: Optional[MetaboliteSetAnalyzer] = None,
    device=None,
    enable_gpt_pubmed: bool = True,
    background_info: Optional[str] = None,
    min_abs_logfc: float = 0.1,
    max_pvalue: float = 0.05,
    min_features: int = 5,
) -> Dict[str, Any]:
    analyzer = _ensure_analyzer(
        analyzer,
        project_root,
        device=device,
        enable_gpt_pubmed=enable_gpt_pubmed,
    )
    workflow = MS2BioTextWorkflow(analyzer=analyzer)
    return workflow.run_set(
        data,
        background_info=background_info,
        min_abs_logfc=min_abs_logfc,
        max_pvalue=max_pvalue,
        min_features=min_features,
    )
