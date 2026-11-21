"""Optional removal of redundant ions (several ions of the same compound) in Mode 2.

In untargeted LC-MS, one compound can appear as several features: different adducts
([M+H]+, [M+Na]+, ...), in-source water loss, dimers and 13C isotopes. When such ions
fall into the same functional module they are counted more than once.

The rule is the one used in the Spec2Function manuscript (Supplementary Methods,
"Redundant-ion analysis"):

* two features are linked when they co-elute within ``rt_tolerance`` seconds (default 5 s)
  and their precursor m/z values back-calculate to the same neutral mass under two of the
  common ion types of the chosen polarity, or differ by a 13C / 13C2 isotope spacing;
  m/z tolerance is max(5 mDa, 10 ppm);
* linked features form compound groups (connected components over the whole input);
* within each functional module, only the feature with the largest MS2 total ion current
  is kept per compound group. Module assignment (clustering and t-SNE) is not changed;
  only the module annotation (retrieval, statistics, LLM naming) uses the retained features.

Removed features stay in the output, flagged ``is_redundant=True`` together with the
``representative_id`` of the feature that was kept for their group.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ISOTOPE_SPACINGS: Dict[str, float] = {"13C": 1.003355, "13C2": 2.006710}

# (ion type, number of M, m/z shift): ion m/z = n * M + shift
COMMON_ION_TYPES: Dict[str, List[Tuple[str, int, float]]] = {
    "positive": [
        ("[M+H]+", 1, 1.007276),
        ("[M+NH4]+", 1, 18.033823),
        ("[M+Na]+", 1, 22.989218),
        ("[M+K]+", 1, 38.963158),
        ("[M+H-H2O]+", 1, -17.003289),
        ("[2M+H]+", 2, 1.007276),
    ],
    "negative": [
        ("[M-H]-", 1, -1.007276),
        ("[M+Cl]-", 1, 34.969402),
        ("[M+HCOO]-", 1, 44.998201),
        ("[M+Na-2H]-", 1, 20.974666),
        ("[M-H2O-H]-", 1, -19.017841),
        ("[2M-H]-", 2, -1.007276),
    ],
}

DEFAULT_RT_TOLERANCE_S = 5.0
DEFAULT_MZ_TOLERANCE_DA = 0.005
DEFAULT_MZ_TOLERANCE_PPM = 10.0

_VID_RE = re.compile(r"mz([\d.]+)\s*rt([\d.]+)", re.IGNORECASE)


def _mz_tol(mz: float, tol_da: float, tol_ppm: float) -> float:
    return max(tol_da, mz * tol_ppm * 1e-6)


def ion_relation(
    mz1: float,
    mz2: float,
    ion_types: Sequence[Tuple[str, int, float]],
    tol_da: float = DEFAULT_MZ_TOLERANCE_DA,
    tol_ppm: float = DEFAULT_MZ_TOLERANCE_PPM,
) -> Optional[str]:
    """Return the relation linking two co-eluting ions of one compound, or None."""
    lo, hi = sorted((mz1, mz2))
    tol = _mz_tol(hi, tol_da, tol_ppm)
    for name, delta in ISOTOPE_SPACINGS.items():
        if abs(hi - lo - delta) <= tol:
            return name
    for a1, n1, s1 in ion_types:
        neutral = (mz1 - s1) / n1
        if neutral <= 0:
            continue
        for a2, n2, s2 in ion_types:
            if a1 == a2:
                continue
            if abs(n2 * neutral + s2 - mz2) <= tol:
                return f"{a1}~{a2}"
    return None


def group_redundant_ions(
    mz: Sequence[float],
    rt: Sequence[float],
    ion_mode: str = "positive",
    rt_tolerance: float = DEFAULT_RT_TOLERANCE_S,
    tol_da: float = DEFAULT_MZ_TOLERANCE_DA,
    tol_ppm: float = DEFAULT_MZ_TOLERANCE_PPM,
) -> Tuple[np.ndarray, List[Tuple[int, int, str]]]:
    """Union-find over co-eluting features with an ion relation.

    Returns compound-group labels (0..n_groups-1) and the list of links (i, j, relation).
    Features with missing m/z or RT form singleton groups.
    """
    ion_mode = str(ion_mode).strip().lower()
    if ion_mode not in COMMON_ION_TYPES:
        raise ValueError(f"ion_mode must be one of {sorted(COMMON_ION_TYPES)}")
    ion_types = COMMON_ION_TYPES[ion_mode]

    mz = np.asarray(mz, dtype=float)
    rt = np.asarray(rt, dtype=float)
    n = len(mz)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    valid = np.where(np.isfinite(mz) & np.isfinite(rt))[0]
    order = valid[np.argsort(rt[valid], kind="stable")]
    links: List[Tuple[int, int, str]] = []
    for a in range(len(order)):
        i = order[a]
        for b in range(a + 1, len(order)):
            j = order[b]
            if rt[j] - rt[i] > rt_tolerance:
                break
            rel = ion_relation(mz[i], mz[j], ion_types, tol_da, tol_ppm)
            if rel:
                links.append((int(i), int(j), rel))
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pj] = pi
    roots = [find(i) for i in range(n)]
    labels = pd.Series(roots).astype("category").cat.codes.to_numpy()
    return labels, links


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def resolve_mz_rt(
    df: pd.DataFrame,
    mz_col: Optional[str] = None,
    rt_col: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, str]]:
    """Find precursor m/z and retention time (seconds) for every row.

    Priority: explicit ``mz_col`` / ``rt_col`` > ``variable_id`` of the form ``mz<m/z>rt<RT>``
    (tidyMass feature IDs, as used in the manuscript) > columns ``mz`` / ``precursor_mz`` and
    ``rt`` / ``rt_sec`` / ``retention_time``. RT is expected in seconds; if every RT is <= 60
    it is taken as minutes and converted.
    """
    source: Dict[str, str] = {}
    vid = None
    if "variable_id" in df.columns:
        parsed = df["variable_id"].astype(str).str.extract(_VID_RE)
        vid = parsed.astype(float)

    has_vid = vid is not None and vid[0].notna().all() and vid[1].notna().all()
    mz_name = mz_col or _find_column(df, ["mz", "m/z", "precursor_mz", "precursormz", "precursor_m/z"])
    if mz_name is None:
        mz_name = next((c for c in df.columns if "precursor" in c.lower()), None)
    if mz_col is None and has_vid:
        mz = vid[0]
        source["mz"] = "variable_id"
    elif mz_name is not None:
        mz = pd.to_numeric(df[mz_name], errors="coerce")
        source["mz"] = f"column '{mz_name}'"
    else:
        mz = pd.Series(np.nan, index=df.index)
        source["mz"] = "missing"

    rt_name = rt_col or _find_column(df, ["rt", "rt_sec", "rt_s", "retention_time", "rtmed"])
    if rt_col is None and has_vid:
        rt = vid[1]
        source["rt"] = "variable_id"
    elif rt_name is not None:
        rt = pd.to_numeric(df[rt_name], errors="coerce")
        source["rt"] = f"column '{rt_name}'"
    else:
        rt = pd.Series(np.nan, index=df.index)
        source["rt"] = "missing"

    if rt.notna().any() and float(rt.max()) <= 60:
        rt = rt * 60.0
        source["rt"] += " (minutes, converted to seconds)"
    mz.index = df.index
    rt.index = df.index
    return mz, rt, source


def mark_redundant_features(
    df: pd.DataFrame,
    valid_idx: Sequence,
    spectra: Sequence[Dict],
    module_col: str = "cluster_id",
    ion_mode: str = "positive",
    rt_tolerance: float = DEFAULT_RT_TOLERANCE_S,
    mz_col: Optional[str] = None,
    rt_col: Optional[str] = None,
) -> Dict:
    """Flag redundant ions within each module, in place.

    Adds ``compound_group``, ``is_redundant`` and ``representative_id`` columns to ``df``
    (rows in ``valid_idx`` only; ``spectra`` are the parsed spectra in the same order, used
    for the MS2 total ion current). Returns a summary dict; if m/z or RT is unavailable the
    step is skipped and nothing is flagged.
    """
    valid_idx = list(valid_idx)
    df["is_redundant"] = False
    df["compound_group"] = np.nan
    df["representative_id"] = None

    summary = {
        "enabled": True,
        "applied": False,
        "rule": "co-elution + common ion types / 13C isotopes; one ion (largest MS2 TIC) kept per compound group within each module",
        "ion_mode": ion_mode,
        "rt_tolerance_s": float(rt_tolerance),
        "mz_tolerance": f"max({DEFAULT_MZ_TOLERANCE_DA * 1000:g} mDa, {DEFAULT_MZ_TOLERANCE_PPM:g} ppm)",
        "ion_types": [name for name, _, _ in COMMON_ION_TYPES.get(str(ion_mode).lower(), [])],
        "n_features": len(valid_idx),
        "n_redundant": 0,
        "n_retained": len(valid_idx),
    }

    sub = df.loc[valid_idx]
    mz, rt, source = resolve_mz_rt(sub, mz_col=mz_col, rt_col=rt_col)
    summary["mz_source"] = source["mz"]
    summary["rt_source"] = source["rt"]
    if mz.isna().all() or rt.isna().all():
        summary["message"] = (
            "Redundancy removal skipped: precursor m/z and retention time are required "
            "(columns 'mz'/'precursor_mz' and 'rt' in seconds, or variable_id like 'mz180.06rt95.2')."
        )
        return summary

    labels, links = group_redundant_ions(mz.to_numpy(), rt.to_numpy(), ion_mode=ion_mode, rt_tolerance=rt_tolerance)
    tic = np.array([float(np.nansum(s.get("intensity", []))) for s in spectra])
    ids = (
        sub["variable_id"].astype(str).to_numpy()
        if "variable_id" in sub.columns
        else np.array([str(i) for i in valid_idx])
    )
    modules = sub[module_col].to_numpy()

    keep = np.ones(len(valid_idx), dtype=bool)
    rep_of = ids.copy()
    tab = pd.DataFrame({"module": modules, "group": labels, "tic": tic, "pos": np.arange(len(valid_idx))})
    for _, g in tab.groupby(["module", "group"], sort=False):
        if len(g) < 2:
            continue
        best = int(g.sort_values("tic", ascending=False, kind="stable").pos.iloc[0])
        for p in g.pos:
            rep_of[p] = ids[best]
            if p != best:
                keep[p] = False

    df.loc[valid_idx, "compound_group"] = labels
    df.loc[valid_idx, "is_redundant"] = ~keep
    df.loc[valid_idx, "representative_id"] = rep_of
    df["is_redundant"] = df["is_redundant"].astype(bool)

    per_module = (
        pd.Series(~keep).groupby(modules).sum().astype(int).to_dict()
    )
    summary.update({
        "applied": True,
        "n_links": len(links),
        "link_types": {str(k): int(v) for k, v in pd.Series([r for _, _, r in links]).value_counts().items()},
        "n_redundant": int((~keep).sum()),
        "n_retained": int(keep.sum()),
        "redundant_per_module": {int(k) + 1: int(v) for k, v in per_module.items()},
    })
    summary["message"] = (
        f"Redundancy removal: {summary['n_redundant']} of {summary['n_features']} features "
        "are additional ions of a compound already represented in the same module and were "
        "excluded from module annotation."
    )
    return summary
