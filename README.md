# Spec2Function

Spec2Function performs MS2 spectrum annotation and metabolite-set functional
analysis using the MS2BioText model. It exposes two entry points:

- `run_single(...)` — annotate a single MS2 spectrum and retrieve similar
  fragments from the bundled HMDB-derived database.
- `run_set(...)` — cluster a set of metabolite spectra (CSV/`DataFrame`),
  generate per-cluster functional summaries, and produce a t-SNE layout.

Large model and reference-database files are hosted on Hugging Face Hub
(`cgxjdzz/ms2function-assets`) and downloaded automatically on first use.

---

## 1. System requirements

### Software dependencies (pinned minimums; tested versions in parentheses)

| Package          | Minimum    | Tested with |
| ---------------- | ---------- | ----------- |
| Python           | 3.10       | 3.10.13     |
| torch            | 2.0        | 2.2.2       |
| transformers     | 4.35       | 4.41.2      |
| numpy            | 1.24       | 1.26.4      |
| pandas           | 2.0        | 2.2.2       |
| scikit-learn     | 1.3        | 1.5.0       |
| huggingface_hub  | 0.20       | 0.23.4      |
| openai           | 1.30       | 1.35.7      |
| python-dotenv    | 1.0        | 1.0.1       |
| tqdm             | 4.65       | 4.66.4      |

All dependencies are installed automatically by `pip install spec2function`.

### Operating systems

Tested on:

- Windows 11 (build 26100)
- Ubuntu 22.04 LTS

The code is OS-agnostic and should also work on macOS 13+.

### Hardware

- Any x86_64 desktop with **≥ 16 GB RAM** and **≥ 5 GB free disk** (for the
  cached model + reference embeddings).
- A CUDA-capable GPU is **optional**. If detected (`torch.cuda.is_available()`),
  the model is moved to GPU automatically; otherwise it runs on CPU.
- Demo and single-spectrum inference run comfortably on CPU. For large set
  analyses (>500 spectra), an NVIDIA GPU with ≥ 6 GB VRAM is recommended.

---

## 2. Installation guide

### From PyPI

```bash
pip install spec2function
```

### From source

```bash
git clone https://github.com/<your-org>/Spec2Function.git
cd Spec2Function
pip install -e .
```

### Typical install time

On a "normal" desktop (4-core CPU, 100 Mbps internet, fresh virtualenv):

- `pip install spec2function`: **2–4 minutes** (most of the time is downloading
  `torch`, ~750 MB).
- First run additionally downloads ~1.2 GB of model + embedding assets from
  Hugging Face Hub: **1–3 minutes** depending on bandwidth. Subsequent runs
  reuse the cached assets and take **no extra download time**.

### Asset cache and environment overrides

| Variable                  | Purpose                                                   |
| ------------------------- | --------------------------------------------------------- |
| `MS2FUNCTION_ASSET_DIR`   | Use a local directory of assets (skip HF download).       |
| `MS2FUNCTION_ASSET_REPO`  | Override the HF repo id (default: `cgxjdzz/ms2function-assets`). |
| `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` | For private asset repos.                       |

Optional LLM / PubMed (only needed for the `annotation` and `story` fields):

| Variable               | Purpose                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| `SILICONFLOW_API_KEY` (or `OPENAI_API_KEY`, `LLM_API_KEY`) | LLM credential. |
| `LLM_PROVIDER`         | `openai`, `siliconflow`, or `gemini` (default: `siliconflow`).   |
| `LLM_MODEL`            | Model name override.                                             |
| `LLM_BASE_URL`         | API base URL override.                                           |
| `PUBMED_EMAIL`         | Contact email for PubMed E-utilities (NCBI requirement).         |

If no LLM key is configured, the workflow still runs end-to-end; LLM-derived
fields fall back to placeholder strings.

---

## 3. Demo

A small bundled demo lives under `examples/`:

- `examples/demo_single.json` — one MS2 spectrum (precursor m/z 184.07, choline-like).
- `examples/demo_set.csv` — 10 features with `logFC` / `pval` / partial annotations.
- `examples/run_demo.py` — runs both entry points and prints a summary.

### How to run

From the repo root:

```bash
python examples/run_demo.py
```

The script disables LLM calls (`enable_gpt_pubmed=False`) so it works without
any API key. To enable LLM annotation, set `SILICONFLOW_API_KEY` (or another
supported key) and remove that flag.

### Expected output

```
============================================================
Demo 1: single-spectrum annotation
============================================================
Top metabolites:        ['Phosphocholine', 'Choline', 'Glycerophosphocholine', ...]
Retrieved fragments:    5 hits
Best hit:               Phosphocholine (similarity=0.87x)
Elapsed:                3.x s

============================================================
Demo 2: metabolite set analysis
============================================================
Clusters discovered:    2
Features in plot_data:  8
  cluster  1: Metabolic Cluster 1                      (known=2, unknown=2)
  cluster  2: Metabolic Cluster 2                      (known=2, unknown=2)
Elapsed:                4.x s
```

The exact metabolite names, similarity values, cluster count, and elapsed times
depend on the asset-pack version, your hardware, and whether assets are already
cached. Variations of ±20% on timing and ±0.05 on similarity scores are normal.

### Expected runtime

On a "normal" desktop (4-core CPU, 16 GB RAM, no GPU), with assets already cached:

- Demo 1 (single spectrum): **2–6 seconds** (dominated by the first model load
  on a cold process; <1 s for repeat calls in the same process).
- Demo 2 (10-feature set): **3–8 seconds**.
- **First-ever** run on a fresh machine adds ~60–180 s for asset download.

GPU runs are typically 2–3× faster after the first call.

---

## 4. Instructions for use

### 4.1 Single spectrum

Input is a JSON-serializable dict with `peaks` (list of `[mz, intensity]`) and
optional `precursor_mz`:

```python
from Spec2Function import run_single

payload = {
    "precursor_mz": 184.0733,
    "peaks": [[60.08, 12.4], [86.10, 38.7], [104.11, 65.1], [184.07, 100.0]],
}

result = run_single(payload, top_k=10, user_focus="lipid metabolism")

print(result["top_metabolites"])      # de-duplicated metabolite names
print(result["retrieved_fragments"])  # list of DB hits with `similarity`
print(result["annotation"])           # LLM summary (if enabled)
print(result["papers"])               # PubMed hits (if enabled)
```

### 4.2 Metabolite set

Input is a CSV path or a `pandas.DataFrame`. The set analysis matches columns
**case-insensitively by substring**:

| Role            | Matches columns containing | Required |
| --------------- | -------------------------- | -------- |
| Spectrum string | `spectrum`                 | Yes      |
| log fold change | `logfc` / `log2fc`         | No (used for filtering & cluster summaries) |
| p-value         | `pval` / `p.value`         | No (used for filtering) |
| Annotation      | `annotator` / `annotation_name` | No (marks known metabolites) |
| Precursor m/z   | `precursor`                | No       |
| Feature ID      | exact column `variable_id` | No (falls back to row index) |

Spectrum-string format: `mz:intensity|mz:intensity|...`
(both `|` and `;` work as separators).

```python
from Spec2Function import run_set

result = run_set(
    "examples/demo_set.csv",
    background_info="case vs control in liver tissue",
    min_abs_logfc=0.5,
    max_pvalue=0.05,
    min_features=5,
)

print(result["story"])         # global functional summary (LLM)
print(result["clusters"])      # per-cluster reports + top metabolites
print(result["plot_data"])     # t-SNE coordinates with cluster labels
print(result["filter"])        # filter info: total/kept/columns used
```

If too few features pass the filters you get
`{"error": "Too few features selected (N)", "filter": {...}}`.

### 4.3 Reusing a workflow (faster for many calls)

If you plan to call inference repeatedly, build the analyzer once:

```python
from Spec2Function import MS2BioTextWorkflow

workflow = MS2BioTextWorkflow.from_spec2function_root(
    project_root=None,           # use the auto-resolved asset cache
    enable_gpt_pubmed=False,
)
single_result = workflow.run_single(payload, top_k=5, include_annotation=False)
set_result = workflow.run_set("examples/demo_set.csv")
```

### 4.4 Visualizing t-SNE

```python
import pandas as pd
import matplotlib.pyplot as plt

plot_df = pd.DataFrame(result["plot_data"])
plt.scatter(plot_df["tsne_x"], plot_df["tsne_y"], c=plot_df["cluster_id"], cmap="tab20")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.show()
```

### 4.5 Troubleshooting

- **Assets not found.** Set `MS2FUNCTION_ASSET_DIR` to a local directory that
  contains `models/best_model.pth`, `models/config.json`,
  `data/hmdb_subsections_WITH_NAME.jsonl`, and `data/all_jsonl_embeddings.pt`,
  or check that `huggingface_hub` is installed and reachable.
- **`annotation` / `story` is a placeholder.** No LLM key was found — set
  `SILICONFLOW_API_KEY` (or another supported key) and rerun.
- **CSV parsing errors.** Make sure your spectrum column contains
  `mz:intensity` pairs separated by `|` or `;`, and the column header includes
  the substring `spectrum`.
- **Too few features.** Relax `min_abs_logfc` / `max_pvalue` or supply more
  rows; the set analysis requires at least 5 features after filtering.

---

## License

See `LICENSE`.
