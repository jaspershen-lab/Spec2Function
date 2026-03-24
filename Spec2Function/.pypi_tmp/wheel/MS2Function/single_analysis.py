from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import hashlib
import json
import os
import pickle
import platform
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel

try:
    from .model.MS2BioText import MS2BioText
    from .model.MSBERT import MSBERT
    from .assets import resolve_assets_root
except Exception as exc:
    raise ImportError("MS2BioText model modules are not available") from exc


if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath


@dataclass
class MS2BioTextAnalysisConfig:
    model_checkpoint: Path
    model_config: Path
    femdb_jsonl: Path
    femdb_embeddings: Path
    specter_cache_dir: Optional[Path] = None
    min_similarity: float = 0.5
    cache_path: Optional[Path] = None
    msbert_vocab_size: int = 100002
    msbert_hidden_size: int = 512
    msbert_num_layers: int = 6
    msbert_num_heads: int = 16
    msbert_dropout: float = 0.0
    msbert_max_len: int = 100
    msbert_kernel_size: int = 3

    @classmethod
    def from_ms2function_root(cls, project_root: Path) -> "MS2BioTextAnalysisConfig":
        root = Path(project_root)
        asset_root = resolve_assets_root(root)
        return cls(
            model_checkpoint=asset_root / "models" / "best_model.pth",
            model_config=asset_root / "models" / "config.json",
            femdb_jsonl=asset_root / "data" / "hmdb_subsections_WITH_NAME.jsonl",
            femdb_embeddings=asset_root / "data" / "all_jsonl_embeddings.pt",
            specter_cache_dir=asset_root / "models" / "specter_cache",
            cache_path=asset_root / "data" / ".index.dat",
        )


def preprocess_spectrum(
    mz: List[float],
    intensity: List[float],
    max_peaks: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    mz = np.array(mz, dtype=np.float32)
    intensity = np.array(intensity, dtype=np.float32)

    if len(intensity) > max_peaks:
        top_indices = np.argsort(intensity)[-max_peaks:]
        mz = mz[top_indices]
        intensity = intensity[top_indices]

    sorted_indices = np.argsort(mz)
    mz = mz[sorted_indices]
    intensity = intensity[sorted_indices]

    if intensity.max() > 0:
        intensity = intensity / intensity.max()

    return mz, intensity


def build_gpt_pubmed_from_ms2function_root(project_root: Path) -> Tuple[object, object]:
    """
    Backwards-compatible helper.

    This used to depend on MS2Function's `backend` module. MS2BioText now ships its own
    `GPTInference` (OpenAI-compatible chat API; defaults to SiliconFlow) and `PubMedSearcher`.

    Configuration is taken from environment variables:
      - SILICONFLOW_API_KEY (recommended) or MS2BIOTEXT_LLM_API_KEY
      - MS2BIOTEXT_LLM_BASE_URL (default: https://api.siliconflow.cn/v1)
      - MS2BIOTEXT_LLM_MODEL (default: Qwen/Qwen2.5-72B-Instruct)
      - MS2BIOTEXT_PUBMED_EMAIL / PUBMED_EMAIL (optional)
    """

    from .llm import GPTInference
    from .pubmed import PubMedSearcher

    try:
        gpt = GPTInference.from_env()
    except ValueError:
        gpt = None
        print("GPT disabled: missing LLM API key.")
    email = os.getenv("MS2BIOTEXT_PUBMED_EMAIL") or os.getenv("PUBMED_EMAIL")
    pubmed = PubMedSearcher(email=email)
    return gpt, pubmed


class SingleSpectrumAnalyzer:
    @classmethod
    def create_from_ms2function_root(
        cls,
        project_root: Path,
        device: Optional[torch.device] = None,
        enable_gpt_pubmed: bool = True,
    ) -> "SingleSpectrumAnalyzer":
        config = MS2BioTextAnalysisConfig.from_ms2function_root(project_root)
        gpt = None
        pubmed = None
        if enable_gpt_pubmed:
            gpt, pubmed = build_gpt_pubmed_from_ms2function_root(project_root)
        return cls(config, device=device, gpt=gpt, pubmed=pubmed)

    def __init__(
        self,
        config: MS2BioTextAnalysisConfig,
        device: Optional[torch.device] = None,
        gpt=None,
        pubmed=None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gpt = gpt
        self.pubmed = pubmed

        self._idx = {"e": {}, "r": {}}
        self._idx_path = Path(config.cache_path) if config.cache_path else None
        if self._idx_path and self._idx_path.exists():
            try:
                self._idx = pickle.load(open(self._idx_path, "rb"))
            except Exception:
                pass

        self._load_model()
        self._load_and_index_data()

    def _load_model(self) -> None:
        with open(self.config.model_config, "r") as f:
            model_config = json.load(f)

        ms_bert = MSBERT(
            self.config.msbert_vocab_size,
            self.config.msbert_hidden_size,
            self.config.msbert_num_layers,
            self.config.msbert_num_heads,
            self.config.msbert_dropout,
            self.config.msbert_max_len,
            self.config.msbert_kernel_size,
        )
        text_model_name = model_config.get("text_model_name") or "allenai/specter"
        text_model_path = model_config.get("text_model_path")
        if text_model_path and Path(text_model_path).exists():
            text_model_source = text_model_path
        else:
            text_model_source = text_model_name

        text_bert = AutoModel.from_pretrained(
            text_model_source,
            cache_dir=self.config.specter_cache_dir,
            local_files_only=True,
        )

        checkpoint = torch.load(
            self.config.model_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        raw_state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in raw_state_dict.items()
        }

        class _LegacyProjectionHead(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
                super().__init__()
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                )

            def forward(self, x):
                return self.projection(x)

        self.model = MS2BioText(ms_bert, text_bert, argparse.Namespace(**model_config))

        if any(
            k.startswith("ms_encoder.projection_head.projection.")
            for k in state_dict.keys()
        ):
            embedding_dim = int(model_config.get("embedding_dim", 512))
            dropout = float(model_config.get("projection_dropout", 0.1))

            self.model.ms_encoder.projection_head = _LegacyProjectionHead(
                self.model.ms_encoder.hidden_size, embedding_dim, dropout
            )
            self.model.text_encoder.projection_head = _LegacyProjectionHead(
                self.model.text_encoder.hidden_size, embedding_dim, dropout
            )
            self.model.text_encoder_base = copy.deepcopy(self.model.text_encoder)
            for p in self.model.text_encoder_base.parameters():
                p.requires_grad = False

        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_and_index_data(self) -> None:
        try:
            cache = torch.load(self.config.femdb_embeddings, weights_only=False)
            self.femdb_embeddings = cache["embeddings"].to("cpu")
            self.femdb_fragments = []
            self.metadata_index = {}
            with open(self.config.femdb_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self.femdb_fragments.append(data)
                        name = data.get("molecule_name")
                        if name:
                            key = name.strip().lower()
                            self.metadata_index.setdefault(key, []).append(data)
                    except Exception:
                        continue
        except Exception:
            self.femdb_embeddings = None
            self.femdb_fragments = []
            self.metadata_index = {}

    def _h(self, mz, intensity, pre) -> str:
        s = f"{float(pre or 0):.2f}|" + "|".join(
            f"{m:.2f}:{i:.2f}" for m, i in zip(mz[:50], intensity[:50])
        )
        return hashlib.md5(s.encode()).hexdigest()

    def _sync(self) -> None:
        if not self._idx_path:
            return
        try:
            self._idx_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self._idx, open(self._idx_path, "wb"))
        except Exception:
            pass

    def encode_ms2(self, mz, intensity, precursor_mz=None) -> torch.Tensor:
        mz, intensity = preprocess_spectrum(mz, intensity)

        word2idx = {"[PAD]": 0, "[MASK]": 1}
        for i, w in enumerate(
            [f"{i:.2f}" for i in np.linspace(0, 1000, 100000, endpoint=False)]
        ):
            word2idx[w] = i + 2

        if precursor_mz is None:
            precursor_mz = 0.0
        precursor_mz = min(float(precursor_mz), 999.99)
        precursor_str = f"{precursor_mz:.2f}"

        peaks_str = [f"{p:.2f}" for p in mz]
        token_ids = [word2idx.get(precursor_str, 0)] + [
            word2idx.get(p, 0) for p in peaks_str
        ]

        intens_seq = np.hstack([2.0, intensity])
        if intens_seq.max() > 0:
            intens_seq = intens_seq / intens_seq.max()

        maxlen = 100
        if len(token_ids) > maxlen:
            token_ids = token_ids[:maxlen]
            intens_seq = intens_seq[:maxlen]
        else:
            n_pad = maxlen - len(token_ids)
            token_ids += [0] * n_pad
            intens_seq = np.hstack([intens_seq, np.zeros(n_pad)])

        mz_tensor = (
            torch.tensor(token_ids, dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
        )
        intensity_tensor = (
            torch.tensor(intens_seq, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(1)
            .to(self.device)
        )

        with torch.no_grad():
            outputs = self.model.encode_ms(mz_tensor, intensity_tensor)
            ms_embed = outputs[0] if isinstance(outputs, tuple) else outputs
            if ms_embed.dim() == 3:
                ms_embed = ms_embed.squeeze(1)

        return ms_embed.cpu().squeeze(0)

    def retrieve_similar_fragments(
        self,
        ms2_embedding: torch.Tensor,
        top_k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[Dict]:
        if ms2_embedding.dim() == 1:
            ms2_embedding = ms2_embedding.unsqueeze(0)

        if self.femdb_embeddings is None or len(self.femdb_embeddings) == 0:
            return []

        db_embeddings = self.femdb_embeddings.to(ms2_embedding.device)
        ms2_norm = F.normalize(ms2_embedding, dim=1)
        db_norm = F.normalize(db_embeddings, dim=1)
        similarities = torch.mm(ms2_norm, db_norm.t())
        if similarities.dim() == 2:
            similarities = similarities.squeeze(0)

        top_similarities, top_indices = torch.topk(
            similarities, k=min(top_k * 2, len(similarities))
        )

        threshold = (
            min_similarity
            if min_similarity is not None
            else self.config.min_similarity
        )

        results = []
        for sim, idx in zip(top_similarities.numpy(), top_indices.numpy()):
            if sim < threshold:
                continue
            fragment = self.femdb_fragments[int(idx)].copy()
            fragment["similarity"] = float(sim)
            results.append(fragment)
            if len(results) >= top_k:
                break

        return results

    def single_inference(
        self,
        mz,
        intensity,
        precursor_mz=None,
        top_k: int = 10,
    ) -> Dict:
        mz_array = np.array(mz, dtype=np.float32)
        intensity_array = np.array(intensity, dtype=np.float32)

        k = self._h(mz_array, intensity_array, precursor_mz)
        if k in self._idx["e"]:
            ms2_embedding = torch.tensor(self._idx["e"][k])
        else:
            ms2_embedding = self.encode_ms2(mz_array, intensity_array, precursor_mz)
            self._idx["e"][k] = ms2_embedding.numpy()

        rk = k + f"_{top_k}"
        if rk in self._idx["r"]:
            retrieved_fragments = self._idx["r"][rk]
        else:
            retrieved_fragments = self.retrieve_similar_fragments(
                ms2_embedding,
                top_k=top_k,
            )
            self._idx["r"][rk] = retrieved_fragments
            self._sync()

        top_metabolites = []
        seen_accessions = set()
        for frag in retrieved_fragments:
            accession = frag.get("accession", "Unknown")
            if accession not in seen_accessions:
                top_metabolites.append(frag.get("molecule_name", "Unknown"))
                seen_accessions.add(accession)

        papers = []
        if self.pubmed and top_metabolites:
            try:
                search_metabolites = top_metabolites[:3]
                papers = self.pubmed.search_by_metabolites(search_metabolites)
            except Exception:
                papers = []

        return {
            "ms2_embedding": ms2_embedding,
            "retrieved_fragments": retrieved_fragments,
            "top_metabolites": top_metabolites,
            "annotation": None,
            "papers": papers,
        }

    def generate_annotation(
        self,
        retrieved_fragments: List[Dict],
        papers: Optional[List[Dict]] = None,
        user_focus: Optional[str] = None,
    ) -> str:
        if not self.gpt:
            return "GPT is disabled. Provide a GPTInference instance to enable it."
        return self.gpt.single_annotation(
            retrieved_fragments=retrieved_fragments,
            papers=papers or [],
            user_focus=user_focus,
        )
