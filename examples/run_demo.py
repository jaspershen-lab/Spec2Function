"""Spec2Function demo script.

Runs both entry points end-to-end on the bundled demo data:

    python examples/run_demo.py

Set `MS2FUNCTION_ASSET_DIR` to point at a local asset directory, or let the
package download assets from Hugging Face Hub on first run.

LLM annotation (`annotation` / `story` fields) requires an API key; if no key
is configured the demo still runs, just with placeholder annotations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from Spec2Function import run_single, run_set

EXAMPLES = Path(__file__).resolve().parent
SINGLE_INPUT = EXAMPLES / "demo_single.json"
SET_INPUT = EXAMPLES / "demo_set.csv"


def demo_single() -> None:
    print("=" * 60)
    print("Demo 1: single-spectrum annotation")
    print("=" * 60)
    payload = json.loads(SINGLE_INPUT.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    result = run_single(payload, top_k=5, enable_gpt_pubmed=False)
    elapsed = time.perf_counter() - t0
    print(f"Top metabolites:        {result.get('top_metabolites', [])[:5]}")
    frags = result.get("retrieved_fragments", [])
    print(f"Retrieved fragments:    {len(frags)} hits")
    if frags:
        top = frags[0]
        print(
            "Best hit:               "
            f"{top.get('molecule_name', 'Unknown')} "
            f"(similarity={top.get('similarity', 0):.3f})"
        )
    print(f"Elapsed:                {elapsed:.1f}s")


def demo_set() -> None:
    print()
    print("=" * 60)
    print("Demo 2: metabolite set analysis")
    print("=" * 60)
    t0 = time.perf_counter()
    result = run_set(
        str(SET_INPUT),
        background_info="demo dataset, case vs control",
        min_abs_logfc=0.5,
        max_pvalue=0.05,
        min_features=5,
        enable_gpt_pubmed=False,
    )
    elapsed = time.perf_counter() - t0
    if "error" in result:
        print(f"Error: {result['error']}")
        print(f"Filter info: {result.get('filter')}")
    else:
        clusters = result.get("clusters", [])
        plot_data = result.get("plot_data", [])
        print(f"Clusters discovered:    {len(clusters)}")
        print(f"Features in plot_data:  {len(plot_data)}")
        for cluster in clusters[:3]:
            print(
                f"  cluster {cluster['id']:>2}: "
                f"{cluster['functional_name']:<40} "
                f"(known={cluster['known_count']}, unknown={cluster['unknown_count']})"
            )
    print(f"Elapsed:                {elapsed:.1f}s")


if __name__ == "__main__":
    demo_single()
    demo_set()
