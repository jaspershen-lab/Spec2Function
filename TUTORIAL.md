# Spec2Function Tutorial

This tutorial walks through a clean, repeatable workflow for MS2 spectrum
annotation (single spectrum) and metabolite set analysis (batch).

## 1) Overview

Spec2Function provides:
- Single-spectrum annotation via MS2BioText embedding + retrieval + optional LLM summary.
- Metabolite set analysis with clustering, functional naming, and a global story.

There are two entry points:
- `run_single(...)` for one spectrum.
- `run_set(...)` for a CSV/`DataFrame` of spectra.

## 2) Install

```bash
pip install spec2function
```

For local development:

```bash
pip install -e .
```

If asset download fails, install Hugging Face Hub:

```bash
pip install huggingface_hub
```

## 3) Assets and Environment

Required assets:
- `models/best_model.pth`
- `models/config.json`
- `data/hmdb_subsections_WITH_NAME.jsonl`
- `data/all_jsonl_embeddings.pt`

Assets are downloaded automatically to a cache directory on first use. You can
override where assets are loaded from:

- `MS2FUNCTION_ASSET_DIR` points to a local asset directory.
- `MS2FUNCTION_ASSET_REPO` overrides the Hugging Face repo (default:
  `cgxjdzz/spec2function-assets`).
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` for private repos.

LLM and PubMed (optional):
- `SILICONFLOW_API_KEY` or `MS2BIOTEXT_LLM_API_KEY`
- `MS2BIOTEXT_LLM_BASE_URL` (default: `https://api.siliconflow.cn/v1`)
- `MS2BIOTEXT_LLM_MODEL` (default: `Qwen/Qwen2.5-72B-Instruct`)
- `MS2BIOTEXT_PUBMED_EMAIL` or `PUBMED_EMAIL`

Disable GPT/PubMed:
- Pass `enable_gpt_pubmed=False` to `run_single`/`run_set`.
- Pass `include_annotation=False` to skip single-spectrum GPT annotation only.

## 4) Data Preparation

### 4.1 Single spectrum JSON

Input is a dict or JSON string with:
- `peaks`: list of `[mz, intensity]` pairs (required)
- `precursor_mz`: float (optional)

Example:

```python
json_input = {
    "peaks": [[100.1, 200.0], [150.2, 300.0], [250.3, 120.0]],
    "precursor_mz": 300.4,
}
```

### 4.2 Metabolite set CSV

Required columns (case-insensitive match by substring):
- Spectrum column: any column containing `spectrum` (example: `ms2_spectrum_string`)

Optional columns:
- `logfc` / `log2fc`: used for filtering and cluster summaries
- `pval` / `p.value`: used for filtering
- `annotator` / `annotation_name`: marks known metabolites
- `precursor`: per-row precursor m/z
- `variable_id`: used in plot output (fallback is row index)

Spectrum string format:
- `mz:intensity|mz:intensity|...` or `mz:intensity;mz:intensity;...`

Example CSV:

```csv
variable_id,ms2_spectrum_string,precursor_mz,logFC,pval,annotation_name
feat_1,"100.1:200|150.2:300|250.3:120",300.4,1.2,0.01,Glucose
feat_2,"110.2:180;175.0:260;260.0:90",320.1,-0.8,0.03,
```

Filtering defaults:
- `min_abs_logfc=0.1`
- `max_pvalue=0.05`
- `min_features=5` (after filtering)

## 5) Single Spectrum Quickstart

```python
from pathlib import Path
from Spec2Function import run_single

json_input = {
    "peaks": [[100.1, 200.0], [150.2, 300.0], [250.3, 120.0]],
    "precursor_mz": 300.4,
}

result = run_single(
    json_input,
    project_root=Path(r"d:\NTU\Spec2Function"),
    top_k=10,
    user_focus="lipid metabolism",
)
print(result.keys())
```

Key outputs:
- `top_metabolites`: de-duplicated metabolite names
- `retrieved_fragments`: database hits with similarity
- `annotation`: LLM summary (unless disabled)
- `papers`: PubMed hits (if enabled)

## 6) Metabolite Set Quickstart

```python
from pathlib import Path
from Spec2Function import run_set

result = run_set(
    r"d:\path\to\your.csv",
    project_root=Path(r"d:\NTU\Spec2Function"),
    background_info="case vs control in liver tissue",
    min_abs_logfc=0.2,
    max_pvalue=0.05,
)
print(result.keys())
```

Key outputs:
- `story`: global functional summary
- `clusters`: per-cluster reports and top metabolites
- `plot_data`: t-SNE coordinates for visualization
- `filter`: filtering info (row counts, chosen columns)

If too few features pass filters, you will get:
- `{"error": "Too few features selected (N)", "filter": {...}}`

## 7) Visualize t-SNE (from run_set)

`run_set` returns `plot_data`, which includes `tsne_x`, `tsne_y`, and `cluster_id`
(0-based). This is ready for scatter plotting.

```python
import pandas as pd
import matplotlib.pyplot as plt

plot_df = pd.DataFrame(result["plot_data"])

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(
    plot_df["tsne_x"],
    plot_df["tsne_y"],
    c=plot_df["cluster_id"],
    cmap="tab20",
    s=30,
    alpha=0.8,
)
ax.set_title("Spec2Function t-SNE clusters")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.show()
```

## 8) Reuse a Workflow (faster for many runs)

```python
from pathlib import Path
from Spec2Function import MS2BioTextWorkflow

workflow = MS2BioTextWorkflow.from_spec2function_root(
    Path(r"d:\NTU\Spec2Function"),
    enable_gpt_pubmed=False,
)

single_result = workflow.run_single(json_input, top_k=5, include_annotation=False)
set_result = workflow.run_set(r"d:\path\to\your.csv")
```

## 9) Troubleshooting

- Missing assets: set `MS2FUNCTION_ASSET_DIR` or ensure the HF download succeeds.
- No GPT output: set API keys or pass `enable_gpt_pubmed=False`.
- CSV parsing errors: verify the spectrum column exists and uses `mz:intensity` pairs.
- Too few features: relax `min_abs_logfc` or `max_pvalue`, or provide more rows.
