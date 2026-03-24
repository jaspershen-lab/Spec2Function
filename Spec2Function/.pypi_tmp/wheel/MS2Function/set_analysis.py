from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from .single_analysis import SingleSpectrumAnalyzer, preprocess_spectrum


class MetaboliteSetAnalyzer(SingleSpectrumAnalyzer):
    @classmethod
    def create_from_ms2function_root(
        cls,
        project_root: Path,
        device: Optional[torch.device] = None,
        enable_gpt_pubmed: bool = True,
    ) -> "MetaboliteSetAnalyzer":
        from .single_analysis import MS2BioTextAnalysisConfig, build_gpt_pubmed_from_ms2function_root

        config = MS2BioTextAnalysisConfig.from_ms2function_root(project_root)
        gpt = None
        pubmed = None
        if enable_gpt_pubmed:
            gpt, pubmed = build_gpt_pubmed_from_ms2function_root(project_root)
        return cls(config, device=device, gpt=gpt, pubmed=pubmed)

    def encode_ms2_batch(self, spectra_list: List[Dict], batch_size: int = 64) -> torch.Tensor:
        word2idx = {"[PAD]": 0, "[MASK]": 1}
        for i, w in enumerate(
            [f"{i:.2f}" for i in np.linspace(0, 1000, 100000, endpoint=False)]
        ):
            word2idx[w] = i + 2

        all_embeddings: List[Optional[torch.Tensor]] = [None] * len(spectra_list)
        to_encode_idx = []
        to_encode_specs = []

        for i, spec in enumerate(spectra_list):
            k = self._h(spec["mz"], spec["intensity"], spec.get("precursor_mz", 0))
            if k in self._idx["e"]:
                all_embeddings[i] = torch.tensor(self._idx["e"][k])
            else:
                to_encode_idx.append(i)
                to_encode_specs.append((spec, k))

        if to_encode_specs:
            for start in range(0, len(to_encode_specs), batch_size):
                end = min(start + batch_size, len(to_encode_specs))
                batch_data = to_encode_specs[start:end]

                try:
                    batch_token_ids = []
                    batch_intensities = []

                    for spec, _ in batch_data:
                        mz, intensity = preprocess_spectrum(spec["mz"], spec["intensity"])
                        precursor = f"{min(float(spec.get('precursor_mz', 0)), 999.99):.2f}"
                        tokens = [word2idx.get(precursor, 0)] + [
                            word2idx.get(f"{p:.2f}", 0) for p in mz
                        ]
                        inten = np.hstack([2.0, intensity])
                        if inten.max() > 0:
                            inten = inten / inten.max()

                        if len(tokens) > 100:
                            tokens = tokens[:100]
                            inten = inten[:100]
                        else:
                            pad = 100 - len(tokens)
                            tokens += [0] * pad
                            inten = np.hstack([inten, np.zeros(pad)])

                        batch_token_ids.append(tokens)
                        batch_intensities.append(inten)

                    mz_t = torch.tensor(batch_token_ids, dtype=torch.long).to(self.device)
                    int_t = torch.tensor(np.array(batch_intensities), dtype=torch.float32).to(
                        self.device
                    )

                    if int_t.dim() == 2:
                        int_t = int_t.unsqueeze(1)

                    with torch.no_grad():
                        outputs = self.model.encode_ms(mz_t, int_t)
                        ms_embeds = outputs[0] if isinstance(outputs, tuple) else outputs
                        if ms_embeds.dim() == 3:
                            ms_embeds = ms_embeds.squeeze(1)
                        ms_embeds = ms_embeds.cpu()

                    for j, (spec, k) in enumerate(batch_data):
                        emb = ms_embeds[j]
                        self._idx["e"][k] = emb.numpy()
                        all_embeddings[to_encode_idx[start + j]] = emb

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception:
                    for j in range(len(batch_data)):
                        all_embeddings[to_encode_idx[start + j]] = torch.zeros(768)

            self._sync()

        return torch.stack(all_embeddings, dim=0)

    def retrieve_function_texts_from_centroid(
        self, centroid: torch.Tensor, top_k: int = 10
    ) -> List[Dict]:
        if centroid.dim() == 1:
            centroid = centroid.unsqueeze(0)

        if self.femdb_embeddings is None or len(self.femdb_embeddings) == 0:
            return []

        db_embeddings = self.femdb_embeddings.to(centroid.device)
        centroid_norm = torch.nn.functional.normalize(centroid, dim=1)
        db_norm = torch.nn.functional.normalize(db_embeddings, dim=1)
        sims = torch.mm(centroid_norm, db_norm.t())
        if sims.dim() == 2:
            sims = sims.squeeze(0)

        top_sims, top_idxs = torch.topk(sims, k=min(top_k * 2, len(sims)))

        hits = {}
        for s, idx in zip(top_sims.numpy(), top_idxs.numpy()):
            frag = self.femdb_fragments[int(idx)]
            name = frag.get("molecule_name", "Unknown")
            if name not in hits:
                hits[name] = {"name": name, "max_similarity": float(s), "texts": []}
            if len(hits[name]["texts"]) < 2:
                text = frag.get("text", "")
                if text:
                    hits[name]["texts"].append(text)

        return sorted(hits.values(), key=lambda x: x["max_similarity"], reverse=True)[
            :top_k
        ]

    def run_semi_supervised_analysis(
        self, df: pd.DataFrame, background_info: Optional[str] = None
    ) -> Dict:
        cols = list(df.columns)
        col_spec = next((c for c in cols if "spectrum" in c.lower()), None)
        col_anno = next(
            (
                c
                for c in cols
                if "annotator" in c.lower() or "annotation_name" in c.lower()
            ),
            None,
        )
        col_fc = next(
            (c for c in cols if "logfc" in c.lower() or "log2fc" in c.lower()),
            None,
        )
        col_mz = next((c for c in cols if "precursor" in c.lower()), None)

        if not col_spec:
            return {"error": "Missing spectrum column (expected 'ms2_spectrum_string' or similar)"}

        parsed = []
        valid_idx = []
        for idx, row in df.iterrows():
            try:
                s = str(row[col_spec]).replace('"', "").strip()
                if not s or s.lower() in ["nan", "none", ""]:
                    continue
                pairs = re.split(r"[|;]", s)
                peaks = [
                    [float(p.split(":")[0]), float(p.split(":")[1])]
                    for p in pairs
                    if ":" in p
                ]
                if peaks:
                    pre = float(row[col_mz]) if col_mz and pd.notna(row[col_mz]) else 0.0
                    parsed.append(
                        {
                            "mz": [p[0] for p in peaks],
                            "intensity": [p[1] for p in peaks],
                            "precursor_mz": pre,
                        }
                    )
                    valid_idx.append(idx)
            except Exception:
                continue

        if not parsed:
            return {"error": f"No valid spectra parsed from {len(df)} rows"}

        if len(parsed) < 5:
            return {"error": f"Too few features selected ({len(parsed)})"}

        try:
            embeds = self.encode_ms2_batch(parsed).cpu().numpy()
        except Exception as exc:
            return {"error": f"Encoding failed: {exc}"}

        try:
            n_components = min(50, embeds.shape[1], len(embeds) - 1)
            min_k = 2
            pca = PCA(n_components=n_components, random_state=42)
            embeds_reduced = pca.fit_transform(embeds)

            max_k = min(20, len(embeds) // 10)
            if max_k < min_k:
                max_k = min_k

            best_score = -1
            best_k = min_k
            for k in range(min_k, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeds_reduced)
                score = silhouette_score(embeds_reduced, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
            labels = kmeans.fit_predict(embeds_reduced)
            df.loc[valid_idx, "cluster_id"] = labels
            n_clusters = best_k

        except Exception as exc:
            return {"error": f"Clustering failed: {exc}"}

        try:
            if len(embeds) > 3:
                perp = min(30, len(embeds) - 1)
                coords = TSNE(
                    n_components=2,
                    perplexity=perp,
                    init="pca",
                    learning_rate="auto",
                    random_state=42,
                ).fit_transform(embeds)
                df.loc[valid_idx, "tsne_x"] = coords[:, 0]
                df.loc[valid_idx, "tsne_y"] = coords[:, 1]
            else:
                df.loc[valid_idx, "tsne_x"] = np.random.rand(len(embeds))
                df.loc[valid_idx, "tsne_y"] = np.random.rand(len(embeds))
        except Exception as exc:
            return {"error": f"t-SNE failed: {exc}"}

        cluster_reports = []
        for cid in range(n_clusters):
            try:
                c_df = df[df["cluster_id"] == cid]
                if len(c_df) == 0:
                    continue

                if col_anno:
                    is_known = c_df[col_anno].notna() & (
                        c_df[col_anno].astype(str).str.len() > 2
                    )
                else:
                    is_known = pd.Series(False, index=c_df.index)
                known_df = c_df[is_known]

                avg_logfc = float(c_df[col_fc].mean()) if col_fc else 0.0
                if avg_logfc >= 0:
                    direction = "Up"
                    direction_label = " Upregulated in Case/Treatment"
                    logfc_interpretation = (
                        "Positive logFC indicates higher abundance in the case/treatment group compared to control"
                    )
                else:
                    direction = "Down"
                    direction_label = " Downregulated in Case/Treatment"
                    logfc_interpretation = (
                        "Negative logFC indicates lower abundance in the case/treatment group compared to control"
                    )

                info = {
                    "id": cid + 1,
                    "known_count": len(known_df),
                    "unknown_count": len(c_df) - len(known_df),
                    "avg_logfc": avg_logfc,
                    "direction": direction,
                    "direction_label": direction_label,
                    "logfc_interpretation": logfc_interpretation,
                    "tsne_center": [
                        float(c_df["tsne_x"].mean()),
                        float(c_df["tsne_y"].mean()),
                    ],
                }

                cluster_mask = df.loc[valid_idx, "cluster_id"] == cid
                cluster_positions = [i for i, mask_val in enumerate(cluster_mask) if mask_val]
                if cluster_positions:
                    centroid = torch.tensor(embeds[cluster_positions]).mean(0)
                else:
                    centroid = None

                if len(known_df) > 0 and col_anno:
                    counts = known_df[col_anno].astype(str).value_counts()
                    top_metabolites = list(counts.head(5).items())
                else:
                    top_metabolites = []

                text_hits = []
                if centroid is not None:
                    text_hits = self.retrieve_function_texts_from_centroid(centroid, top_k=10)

                if self.gpt:
                    try:
                        functional_name, theme_prompt = self.gpt.generate_cluster_functional_name(
                            cluster_stats=info,
                            top_metabolites=top_metabolites,
                            retrieved_texts=text_hits,
                            background_info=background_info,
                            debug_output_path=None,
                            return_prompt=True,
                        )
                    except Exception:
                        functional_name = f"Metabolic Cluster {cid+1}"
                        theme_prompt = None
                else:
                    functional_name = f"Metabolic Cluster {cid+1}"
                    theme_prompt = None

                if self.gpt:
                    try:
                        cluster_report = self.gpt.generate_cluster_report(
                            cluster_stats=info,
                            functional_name=functional_name,
                            top_metabolites=top_metabolites,
                            retrieved_texts=text_hits,
                            example_logfc=c_df[col_fc].tolist() if col_fc else None,
                            background_info=background_info,
                        )
                    except Exception:
                        cluster_report = f"Report generation failed for {functional_name}."
                else:
                    cluster_report = "GPT is disabled. Provide a GPTInference instance to enable it."

                papers = []
                if len(c_df) > 2 and self.pubmed:
                    try:
                        if self.gpt:
                            pubmed_query = self.gpt.generate_pubmed_query(
                                functional_name=functional_name,
                                top_metabolites=top_metabolites,
                                background_info=background_info,
                            )
                        else:
                            pubmed_query = top_metabolites[0][0] if top_metabolites else functional_name

                        papers = self.pubmed.search_by_metabolites([pubmed_query], max_results=3)
                    except Exception:
                        papers = []

                cluster_info = {
                    "id": cid + 1,
                    "functional_name": functional_name,
                    "report": cluster_report,
                    "type": "Confirmed" if len(known_df) > 0 else "Inferred",
                    "known_count": info["known_count"],
                    "unknown_count": info["unknown_count"],
                    "avg_logfc": info["avg_logfc"],
                    "direction": info["direction"],
                    "direction_label": info["direction_label"],
                    "tsne_center": info["tsne_center"],
                    "top_metabolites": [name for name, _ in top_metabolites],
                    "papers": papers,
                    "gpt_theme_prompt": theme_prompt,
                }
                cluster_reports.append(cluster_info)

            except Exception:
                continue

        cluster_reports.sort(key=lambda x: abs(x["avg_logfc"]), reverse=True)

        if self.gpt:
            try:
                story = self.gpt.generate_global_story(
                    cluster_reports=cluster_reports, background_info=background_info
                )
            except Exception as exc:
                story = f"GPT generation failed: {exc}"
        else:
            story = "GPT is disabled. Provide a GPTInference instance to enable it."

        plot_df = df.loc[valid_idx].copy()
        if col_anno:
            plot_df["Annotator"] = plot_df[col_anno]
        else:
            plot_df["Annotator"] = "NA"

        if "variable_id" not in plot_df.columns:
            plot_df["variable_id"] = plot_df.index.astype(str)

        return {
            "story": story,
            "clusters": cluster_reports,
            "plot_data": plot_df[
                ["variable_id", "tsne_x", "tsne_y", "cluster_id", "Annotator"]
            ].to_dict("records"),
        }
