# -*- coding: utf-8 -*-
"""
Spec2Function Model Manager - Core Backend
Refactored with:
  1. GMM+BIC for automatic cluster selection
  2. Centroid-based functional text retrieval
  3. Two-tier GPT architecture for cluster theme generation
"""
import torch
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
import re
import hashlib
import pickle
from typing import List, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from collections import Counter, defaultdict
from pathlib import Path
import sys
import pathlib
import platform
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

from transformers import AutoTokenizer, BertModel

from .model.MS2BioText import MS2BioText
from .model.MSBERT import MSBERT

from .config import Spec2FunctionConfig
from .gpt_inference import GPTInference
from .pubmed import PubMedSearcher
from .utils import preprocess_spectrum

class Spec2FunctionManager:
    _instance = None
    
    def __new__(cls, config: Optional[Spec2FunctionConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self, config: Optional[Spec2FunctionConfig] = None):
        if self._initialized: return
        print("Initializing Backend...")
        from .config import config as default_config
        self.config = config if config else default_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
        self._load_and_index_data()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.specter_model_name,
            cache_dir=self.config.specter_cache_dir,
            local_files_only=False
        )
        
        # Note: translated comment removed.
        llm_api_key = self.config.resolve_llm_api_key()
        if llm_api_key:
            self.gpt = GPTInference(
                api_key=llm_api_key,
                model=self.config.llm_model,
                max_tokens=self.config.gpt_max_tokens,
                temperature=self.config.gpt_temperature,
                provider=self.config.llm_provider,
                base_url=self.config.llm_base_url
            )
        else:
            print(" GPT disabled: LLM API key not set")
            self.gpt = None
            
        self.pubmed = PubMedSearcher(email=self.config.pubmed_email)
        
        self._idx = {'e': {}, 'r': {}}
        self._idx_path = Path(__file__).resolve().parent.parent / "data" / ".index.dat"
        try:
            if self._idx_path.exists():
                self._idx = pickle.load(open(self._idx_path, 'rb'))
        except: pass
        
        self._initialized = True

    def _h(self, mz, intensity, pre):
        s = f"{float(pre or 0):.2f}|" + "|".join(f"{m:.2f}:{i:.2f}" for m, i in zip(mz[:50], intensity[:50]))
        return hashlib.md5(s.encode()).hexdigest()
    
    def _sync(self):
        try:
            self._idx_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self._idx, open(self._idx_path, 'wb'))
        except: pass

    def update_llm_config(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        if not hasattr(self, "config"):
            from .config import config as default_config
            self.config = default_config

        provider = (provider or "openai").lower()
        self.config.llm_provider = provider
        resolved_model = (model or "").strip()
        resolved_base_url = (base_url or "").strip()

        if provider == "siliconflow":
            if not resolved_model:
                resolved_model = "deepseek-ai/DeepSeek-V3.2"
            if not resolved_base_url:
                resolved_base_url = "https://api.siliconflow.cn/v1/chat/completions"
        elif provider == "openai" and not resolved_model:
            resolved_model = self.config.gpt_model

        if resolved_model:
            self.config.llm_model = resolved_model
        if resolved_base_url:
            self.config.llm_base_url = resolved_base_url

        if api_key:
            self.config.llm_api_key = api_key.strip()
            if provider == "gemini":
                self.config.gemini_api_key = api_key.strip()
            elif provider == "siliconflow":
                self.config.siliconflow_api_key = api_key.strip()
            else:
                self.config.openai_api_key = api_key.strip()

        if temperature is not None:
            self.config.gpt_temperature = float(temperature)
        if max_tokens is not None:
            self.config.gpt_max_tokens = int(max_tokens)

        llm_api_key = self.config.resolve_llm_api_key()
        if not llm_api_key:
            self.gpt = None
            return "LLM disabled: missing API key."

        self.gpt = GPTInference(
            api_key=llm_api_key,
            model=self.config.llm_model,
            max_tokens=self.config.gpt_max_tokens,
            temperature=self.config.gpt_temperature,
            provider=self.config.llm_provider,
            base_url=self.config.llm_base_url
        )
        return f"LLM updated: provider={self.config.llm_provider}, model={self.config.llm_model}."

    def _load_model(self):
        with open(self.config.model_config, 'r') as f:
            model_config = json.load(f)
        ms_bert = MSBERT(
            self.config.msbert_vocab_size, 
            self.config.msbert_hidden_size, 
            self.config.msbert_num_layers, 
            self.config.msbert_num_heads, 
            self.config.msbert_dropout, 
            self.config.msbert_max_len, 
            self.config.msbert_kernel_size
        )
        text_bert = BertModel.from_pretrained(
            'allenai/specter',
            cache_dir=self.config.specter_cache_dir,
            local_files_only=False
        )
        import argparse
        self.model = MS2BioText(ms_bert, text_bert, argparse.Namespace(**model_config))
        checkpoint = torch.load(self.config.model_checkpoint, map_location='cpu', weights_only=False)
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_and_index_data(self):
        try:
            cache = torch.load(self.config.femdb_embeddings, weights_only=False)
            self.femdb_embeddings = cache['embeddings'].to('cpu')
            self.femdb_fragments = []
            self.metadata_index = defaultdict(list)
            with open(self.config.femdb_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self.femdb_fragments.append(data)
                        if 'molecule_name' in data:
                            self.metadata_index[data['molecule_name'].strip().lower()].append(data)
                    except: continue
            print(f"OK Indexed {len(self.metadata_index)} molecules.")
        except:
            print(" Failed to load JSONL.")
            self.femdb_fragments = []

    def _get_molecule_profile(self, name):
        entries = self.metadata_index.get(name.strip().lower(), [])
        if not entries: return f"No profile for {name}."
        cats = defaultdict(list)
        for e in entries: cats[e.get('type', 'general')].append(e.get('text', ''))
        profile = f"**{name} Profile:**\n"
        for t in ['pathways', 'diseases', 'tissue_locations']:
            if t in cats: profile += f"- {t}: {'; '.join(list(set(cats[t]))[:3])}\n"
        return profile

    # ============================================
    # Single Spectrum Inference Functions
    # ============================================
    
    def encode_ms2(self, mz, intensity, precursor_mz=None):
        #Note: translated content removed.
        mz, intensity = preprocess_spectrum(mz, intensity)
        
        # Note: translated content removed.
        word2idx = {'[PAD]': 0, '[MASK]': 1}
        for i, w in enumerate([f"{i:.2f}" for i in np.linspace(0, 1000, 100000, endpoint=False)]):
            word2idx[w] = i + 2
        
        # Note: translated content removed.
        if precursor_mz is None:
            precursor_mz = 0.0
        precursor_mz = min(float(precursor_mz), 999.99)
        precursor_str = f"{precursor_mz:.2f}"
        
        # Tokenization
        peaks_str = [f"{p:.2f}" for p in mz]
        token_ids = [word2idx.get(precursor_str, 0)] + [word2idx.get(p, 0) for p in peaks_str]
        
        # Intensity normalization
        intens_seq = np.hstack([2.0, intensity])
        if intens_seq.max() > 0:
            intens_seq = intens_seq / intens_seq.max()
        
        # Padding/Truncating
        maxlen = 100
        if len(token_ids) > maxlen:
            token_ids = token_ids[:maxlen]
            intens_seq = intens_seq[:maxlen]
        else:
            n_pad = maxlen - len(token_ids)
            token_ids += [0] * n_pad
            intens_seq = np.hstack([intens_seq, np.zeros(n_pad)])
        
        # Convert to tensors
        mz_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        intensity_tensor = torch.tensor(intens_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model.encode_ms(mz_tensor, intensity_tensor)
            
            #Note: translated content removed.
            if isinstance(outputs, tuple):
                ms_embed = outputs[0]
            else:
                ms_embed = outputs
            
            #Note: translated content removed.
            if ms_embed.dim() == 3:
                ms_embed = ms_embed.squeeze(1)
        
        return ms_embed.cpu().squeeze(0)

    def retrieve_similar_fragments(self, ms2_embedding: torch.Tensor, top_k: int = None, min_similarity: float = None) -> List[Dict]:
        """FemDB fragments - Single mode"""
        if ms2_embedding.dim() == 1:
            ms2_embedding = ms2_embedding.unsqueeze(0)

        if not hasattr(self, "femdb_embeddings") or self.femdb_embeddings is None or len(self.femdb_embeddings) == 0:
            return []

        # 使用 config 中的默认值
        if top_k is None:
            top_k = getattr(self.config, 'single_retrieval_top_k', 10)
        if min_similarity is None:
            min_similarity = getattr(self.config, 'min_similarity', 0.5)

        db_embeddings = self.femdb_embeddings.to(ms2_embedding.device)
        ms2_norm = F.normalize(ms2_embedding, dim=1)
        db_norm = F.normalize(db_embeddings, dim=1)
        similarities = torch.mm(ms2_norm, db_norm.t())
        
        if similarities.dim() == 2:
            similarities = similarities.squeeze(0)
        
        top_similarities, top_indices = torch.topk(similarities, k=min(top_k * 2, len(similarities)))
        
        results = []
        for sim, idx in zip(top_similarities.numpy(), top_indices.numpy()):
            if sim < min_similarity:
                continue
            
            fragment = self.femdb_fragments[int(idx)].copy()
            fragment['similarity'] = float(sim)
            results.append(fragment)
            
            if len(results) >= top_k:
                break
        
        return results

    def single_inference(self, mz, intensity, precursor_mz=None, top_k=None, user_focus=None):
        """
        MS2 spectrumOK         """
        print(f" Processing single spectrum with {len(mz)} peaks...")
        annotation = None
        papers = []
        
        mz_array = np.array(mz, dtype=np.float32)
        intensity_array = np.array(intensity, dtype=np.float32)
        if top_k is None:
            top_k = getattr(self.config, 'single_retrieval_top_k', 10)
        k = self._h(mz_array, intensity_array, precursor_mz)
        
        if k in self._idx['e']:
            ms2_embedding = torch.tensor(self._idx['e'][k])
        else:
            ms2_embedding = self.encode_ms2(mz_array, intensity_array, precursor_mz)
            self._idx['e'][k] = ms2_embedding.numpy()
        print(f"  OK Encoded to shape {ms2_embedding.shape}")
        
        rk = k + f"_{top_k}"
        if rk in self._idx['r']:
            retrieved_fragments = self._idx['r'][rk]
        else:
            retrieved_fragments = self.retrieve_similar_fragments(
                ms2_embedding,
                top_k=top_k,
                min_similarity=self.config.min_similarity if hasattr(self.config, 'min_similarity') else 0.5
            )
            self._idx['r'][rk] = retrieved_fragments
            self._sync()
        print(f"  OK Retrieved {len(retrieved_fragments)} fragments")
        
        top_metabolites = []
        seen_accessions = set()
        for frag in retrieved_fragments:
            accession = frag.get('accession', 'Unknown')
            if accession not in seen_accessions:
                top_metabolites.append(frag.get('molecule_name', 'Unknown'))
                seen_accessions.add(accession)
        
        if len(top_metabolites) >= 1:
            search_metabolites = top_metabolites[:3]
            print(f"   Searching PubMed for: {search_metabolites}")
            
            try:
                papers = self.pubmed.search_by_metabolites(
                    search_metabolites,
                    max_results=self.config.pubmed_max_results if hasattr(self.config, 'pubmed_max_results') else 10
                )
                print(f"  OK Found {len(papers)} papers")
            except Exception as e:
                print(f"   PubMed search failed: {e}")
        
        if 'annotation' not in locals():
            annotation = None
        if 'papers' not in locals():
            papers = []

        result = {
            'ms2_embedding': ms2_embedding,
            'retrieved_fragments': retrieved_fragments,
            'top_metabolites': top_metabolites,
            'annotation': annotation,
            'papers': papers
        }
        
        print(f"OK Single inference complete!")
        return result

    # ============================================
    # Batch Encoding (for Set Mode)
    # ============================================

    def encode_ms2_batch(self, spectra_list: List[Dict], batch_size: int = 64) -> torch.Tensor:
        """
        [] OK         """
        print(f"   Encoding {len(spectra_list)} spectra in batches of {batch_size}...")
        
        # Note: translated comment removed.
        word2idx = {'[PAD]': 0, '[MASK]': 1}
        for i, w in enumerate([f"{i:.2f}" for i in np.linspace(0, 1000, 100000, endpoint=False)]):
            word2idx[w] = i + 2
        
        all_embeddings = [None] * len(spectra_list)
        to_encode_idx = []
        to_encode_specs = []
        
        for i, spec in enumerate(spectra_list):
            k = self._h(spec['mz'], spec['intensity'], spec.get('precursor_mz', 0))
            if k in self._idx['e']:
                all_embeddings[i] = torch.tensor(self._idx['e'][k])
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
                        mz, intensity = preprocess_spectrum(spec['mz'], spec['intensity'])
                        precursor = f"{min(float(spec.get('precursor_mz', 0)), 999.99):.2f}"
                        tokens = [word2idx.get(precursor, 0)] + [word2idx.get(f"{p:.2f}", 0) for p in mz]
                        inten = np.hstack([2.0, intensity])
                        if inten.max() > 0:
                            inten /= inten.max()
                        
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
                    int_t = torch.tensor(np.array(batch_intensities), dtype=torch.float32).to(self.device)
                    
                    if int_t.dim() == 2:
                        int_t = int_t.unsqueeze(1)
                    
                    with torch.no_grad():
                        outputs = self.model.encode_ms(mz_t, int_t)
                        
                        if isinstance(outputs, tuple):
                            ms_embeds = outputs[0]
                        else:
                            ms_embeds = outputs
                        
                        if ms_embeds.dim() == 3:
                            ms_embeds = ms_embeds.squeeze(1)
                        
                        ms_embeds = ms_embeds.cpu()
                    
                    for j, (spec, k) in enumerate(batch_data):
                        emb = ms_embeds[j]
                        self._idx['e'][k] = emb.numpy()
                        all_embeddings[to_encode_idx[start + j]] = emb
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if (start // batch_size + 1) % 10 == 0 or end == len(to_encode_specs):
                        print(f"    OK Processed {end}/{len(to_encode_specs)} new spectra")
                        
                except Exception as e:
                    print(f"OK Failed to encode batch {start}-{end}: {e}")
                    import traceback
                    traceback.print_exc()
                    for j in range(len(batch_data)):
                        all_embeddings[to_encode_idx[start + j]] = torch.zeros(768)
            
            self._sync()
        
        final_embeddings = torch.stack(all_embeddings, dim=0)
        print(f"  OK Encoding complete: {final_embeddings.shape}")
        
        return final_embeddings

    # ============================================
    # NEW: Functional Text Retrieval
    # ============================================
    
    def retrieve_function_texts_from_centroid(self, 
                                            centroid: torch.Tensor, 
                                            top_k: int = None,
                                            min_similarity: float = None) -> List[Dict]:
        """
        从 centroid 检索功能文本
        """
        if centroid.dim() == 1:
            centroid = centroid.unsqueeze(0)

        if not hasattr(self, "femdb_embeddings") or self.femdb_embeddings is None or len(self.femdb_embeddings) == 0:
            return []
        
        # 使用 config 中的默认值
        if top_k is None:
            top_k = getattr(self.config, 'set_retrieval_top_k', 5)
        if min_similarity is None:
            min_similarity = getattr(self.config, 'min_similarity', 0.5)
        
        db_embeddings = self.femdb_embeddings.to(centroid.device)
        centroid_norm = F.normalize(centroid, dim=1)
        db_norm = F.normalize(db_embeddings, dim=1)
        sims = torch.mm(centroid_norm, db_norm.t())
        if sims.dim() == 2:
            sims = sims.squeeze(0)
        
        top_sims, top_idxs = torch.topk(sims, k=min(top_k * 3, len(sims)))  # 多取一些备用
        
        print(f"       ✓ Retrieved {len(top_idxs)} raw fragments from FemDB")
        
        fragments_processed = 0
        filtered_count = 0
        hits = {}
        
        for s, idx in zip(top_sims.numpy(), top_idxs.numpy()):
            # min_similarity 过滤
            if s < min_similarity:
                filtered_count += 1
                continue
                
            frag = self.femdb_fragments[int(idx)]
            name = frag.get('molecule_name', 'Unknown')
            
            if name not in hits:
                hits[name] = {
                    'name': name,
                    'max_similarity': float(s),
                    'texts': []
                }
            
            if len(hits[name]['texts']) < 2:
                text = frag.get('text', '')
                if text:
                    hits[name]['texts'].append(text)
                    fragments_processed += 1
        
        results = sorted(hits.values(), key=lambda x: x['max_similarity'], reverse=True)[:top_k]
        
        print(f"       ✓ Filtered {filtered_count} fragments below min_similarity={min_similarity}")
        print(f"       ✓ Aggregated into {len(hits)} unique metabolites")
        print(f"       ✓ Returning top {len(results)} metabolites")
        
        return results

    def retrieve_function_texts_by_voting(self,
                                          embeddings: torch.Tensor,
                                          top_k_per_sample: int = 5,
                                          final_top_k: int = 5,
                                          min_similarity: float = None) -> List[Dict]:
        """
        对cluster内每个MS2都检索top_k，然后统计频数，返回排名最高的final_top_k个
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        if not hasattr(self, "femdb_embeddings") or self.femdb_embeddings is None or len(self.femdb_embeddings) == 0:
            return []
        
        if min_similarity is None:
            min_similarity = getattr(self.config, 'min_similarity', 0.5)
        
        print(f"       ✓ Voting retrieval: {embeddings.shape[0]} samples, top_k_per_sample={top_k_per_sample}")
        
        db_embeddings = self.femdb_embeddings.to(embeddings.device)
        embeddings_norm = F.normalize(embeddings, dim=1)
        db_norm = F.normalize(db_embeddings, dim=1)
        
        # 计算所有sample和所有db的相似度矩阵 [n_samples, n_db]
        all_sims = torch.mm(embeddings_norm, db_norm.t())
        
        # 统计每个metabolite被检索到的频数和最高相似度
        metabolite_votes = defaultdict(lambda: {'count': 0, 'max_sim': 0.0, 'texts': []})
        
        for sample_idx in range(all_sims.shape[0]):
            sims = all_sims[sample_idx]
            top_sims, top_idxs = torch.topk(sims, k=min(top_k_per_sample * 2, len(sims)))
            
            seen_in_sample = set()
            count_for_sample = 0
            
            for sim_val, idx in zip(top_sims.numpy(), top_idxs.numpy()):
                if sim_val < min_similarity:
                    continue
                if count_for_sample >= top_k_per_sample:
                    break
                    
                frag = self.femdb_fragments[int(idx)]
                name = frag.get('molecule_name', 'Unknown')
                
                if name not in seen_in_sample:
                    seen_in_sample.add(name)
                    metabolite_votes[name]['count'] += 1
                    metabolite_votes[name]['max_sim'] = max(metabolite_votes[name]['max_sim'], float(sim_val))
                    
                    # 收集文本
                    text = frag.get('text', '')
                    if text and len(metabolite_votes[name]['texts']) < 2:
                        metabolite_votes[name]['texts'].append(text)
                    
                    count_for_sample += 1
        
        # 按频数排序，频数相同则按最高相似度排序
        sorted_metabolites = sorted(
            metabolite_votes.items(),
            key=lambda x: (x[1]['count'], x[1]['max_sim']),
            reverse=True
        )
        
        print(f"       ✓ Found {len(sorted_metabolites)} unique metabolites across all samples")
        
        # 构建返回结果
        results = []
        for name, info in sorted_metabolites[:final_top_k]:
            results.append({
                'name': name,
                'max_similarity': info['max_sim'],
                'vote_count': info['count'],
                'texts': info['texts']
            })
        
        if results:
            print(f"       ✓ Top {len(results)} by voting:")
            for i, r in enumerate(results[:3], 1):
                print(f"         {i}. {r['name']} (votes={r['vote_count']}, max_sim={r['max_similarity']:.4f})")
        
        return results

    # ============================================
    # Semi-Supervised Analysis (Set Mode) - REFACTORED
    # ============================================

    def run_semi_supervised_analysis(self, df: pd.DataFrame, background_info: str = None,
                                       retrieval_method: str = 'centroid', top_k: int = 5,
                                        min_similarity: float = 0.65):
        """
        Note: translated content removed.
        retrieval_method: 'centroid' or 'voting'
        top_k: number of top results to retrieve
        """
        print(f" Starting analysis on {len(df)} rows...")
        print(f"   Retrieval method: {retrieval_method}, top_k: {top_k}")
        
        try:
            # Note: translated comment removed.
            cols = list(df.columns)
            col_spec = next((c for c in cols if 'spectrum' in c.lower()), None)
            col_anno = next((c for c in cols if 'Annotator' in c or 'Annotation_Name' in c), None)
            col_fc = next((c for c in cols if 'logFC' in c or 'log2fc' in c.lower()), None)
            col_mz = next((c for c in cols if 'precursor' in c.lower()), None)
            col_score = next((c for c in cols if 'ms2_score' in c.lower() or c.lower() == 'score' or 'match_score' in c.lower()), None)

            if not col_spec:
                return {"error": "Missing spectrum column (expected 'ms2_spectrum_string' or similar)"}

            print(f" Columns detected: spectrum={col_spec}, anno={col_anno}, fc={col_fc}, mz={col_mz}, score={col_score}")

            # Note: translated comment removed.
            print(" Parsing spectra...")
            parsed, valid_idx = [], []
            for idx, row in df.iterrows():
                try:
                    s = str(row[col_spec]).replace('"', '').strip()
                    if not s or s.lower() in ['nan', 'none', '']:
                        continue
                        
                    pairs = re.split(r'[|;]', s)
                    peaks = [[float(p.split(':')[0]), float(p.split(':')[1])] for p in pairs if ':' in p]
                    
                    if peaks:
                        pre = float(row[col_mz]) if col_mz and pd.notna(row[col_mz]) else 0.0
                        parsed.append({
                            'mz': [p[0] for p in peaks],
                            'intensity': [p[1] for p in peaks],
                            'precursor_mz': pre
                        })
                        valid_idx.append(idx)
                except Exception as e:
                    continue
                    
            if not parsed:
                return {"error": f"No valid spectra parsed from {len(df)} rows"}
            
            print(f"OK Parsed {len(parsed)}/{len(df)} spectra successfully")
            
            # Note: translated comment removed.
            print(" Encoding spectra with MS2BioText model...")
            try:
                embeds = self.encode_ms2_batch(parsed).cpu().numpy()
                print(f"OK Encoded to shape {embeds.shape}")
            except Exception as e:
                print(f"OK Encoding failed: {e}")
                import traceback
                traceback.print_exc()
                return {"error": f"Encoding failed: {str(e)}"}
            
            # Note: translated comment removed.
            print(" Running PCA + K-means clustering...")
            try:
                n_components = min(50, embeds.shape[1], len(embeds) - 1)
                min_k = 2
                # Note: translated comment removed.
                print(f"  OK PCA: reducing from {embeds.shape[1]}D to {n_components}D...")
                pca = PCA(n_components=n_components, random_state=42)
                embeds_reduced = pca.fit_transform(embeds)
                explained_var = pca.explained_variance_ratio_.sum()
                print(f"    OK Explained variance: {explained_var*100:.1f}%")
                
                # Note: translated comment removed.
                max_k = min(20, len(embeds) // 10)
                if max_k < min_k:
                    max_k = min_k
                
                # Note: translated comment removed.
                print(f"  OK Testing K from {min_k} to {max_k} (silhouette score)...")
                silhouette_scores = []
                best_score = -1
                best_k = min_k
                
                for k in range(min_k, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeds_reduced)
                    score = silhouette_score(embeds_reduced, labels)
                    silhouette_scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                    
                    if k % 5 == 0 or k == max_k:
                        print(f"    K={k}: silhouette={score:.3f}")
                
                # Note: translated comment removed.
                print(f"  OK Using optimal K={best_k} (silhouette={best_score:.3f})")
                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
                labels = kmeans.fit_predict(embeds_reduced)
                df.loc[valid_idx, 'cluster_id'] = labels
                n_clusters = best_k
                
                print(f"OK Clustering complete: {n_clusters} clusters")
                
            except Exception as e:
                print(f"OK Clustering failed: {e}")
                import traceback
                traceback.print_exc()
                return {"error": f"Clustering failed: {str(e)}"}
            
            # Note: translated comment removed.
            print(" Running t-SNE dimensionality reduction...")
            try:
                if len(embeds) > 3:
                    perp = min(30, len(embeds) - 1)
                    coords = TSNE(
                        n_components=2, 
                        perplexity=perp, 
                        init='pca', 
                        learning_rate='auto', 
                        random_state=42
                    ).fit_transform(embeds)
                    df.loc[valid_idx, 'tsne_x'] = coords[:, 0]
                    df.loc[valid_idx, 'tsne_y'] = coords[:, 1]
                else:
                    df.loc[valid_idx, 'tsne_x'] = np.random.rand(len(embeds))
                    df.loc[valid_idx, 'tsne_y'] = np.random.rand(len(embeds))
                print(f"OK t-SNE complete")
            except Exception as e:
                print(f"OK t-SNE failed: {e}")
                return {"error": f"t-SNE failed: {str(e)}"}

            # Note: translated comment removed.
            print("=" * 80)
            cluster_reports = []
            
            for cid in range(n_clusters):
                try:
                    c_df = df[df['cluster_id'] == cid]
                    if len(c_df) == 0:
                        continue
                    
                    print(f"\n CLUSTER {cid+1} Analysis:")
                    print("-" * 80)
                    
                    # Note: translated comment removed.
                    if col_anno:
                        is_known = c_df[col_anno].notna() & (c_df[col_anno].astype(str).str.len() > 2)
                    else:
                        is_known = pd.Series(False, index=c_df.index)
                    known_df = c_df[is_known]
                    
                    avg_logfc = float(c_df[col_fc].mean()) if col_fc else 0.0
                    
                    # Note: translated comment removed.
                    if avg_logfc >= 0:
                        direction = "Up"
                        direction_label = " Upregulated in Case/Treatment"
                        logfc_interpretation = "Positive logFC indicates higher abundance in the case/treatment group compared to control"
                    else:
                        direction = "Down"
                        direction_label = " Downregulated in Case/Treatment"
                        logfc_interpretation = "Negative logFC indicates lower abundance in the case/treatment group compared to control"
                    
                    info = {
                        "id": cid + 1,
                        "known_count": len(known_df),
                        "unknown_count": len(c_df) - len(known_df),
                        "avg_logfc": avg_logfc,
                        "direction": direction,
                        "direction_label": direction_label,
                        "logfc_interpretation": logfc_interpretation,  # 
                        "tsne_center": [float(c_df['tsne_x'].mean()), float(c_df['tsne_y'].mean())]
                    }
                    
                    print(f"   Basic Stats:")
                    print(f"     - Size: {len(c_df)} metabolites ({info['known_count']} known, {info['unknown_count']} unknown)")
                    print(f"     - Trend: {direction_label} (avg logFC: {avg_logfc:.3f})")
                    print(f"     - Note: {logfc_interpretation}")
                    
                    # Note: translated comment removed.
                    cluster_mask = df.loc[valid_idx, 'cluster_id'] == cid
                    cluster_positions = [i for i, mask_val in enumerate(cluster_mask) if mask_val]
                    
                    if cluster_positions:
                        centroid = torch.tensor(embeds[cluster_positions]).mean(0)
                        print(f"     - Centroid computed from {len(cluster_positions)} embeddings")
                    else:
                        centroid = None
                        print(f"      No centroid (empty cluster)")
                    
                    # Note: translated comment removed.
                    if len(known_df) > 0 and col_anno:
                        counts = known_df[col_anno].astype(str).value_counts()
                        top_names = list(counts.head(5).index)
                        if col_score:
                            avg_scores = known_df.groupby(known_df[col_anno].astype(str))[col_score].mean()
                            top_metabolites = [(name, counts[name], float(avg_scores.get(name, 0.0))) for name in top_names]
                        else:
                            top_metabolites = [(name, counts[name], None) for name in top_names]
                        print(f"\n   Top Known Metabolites:")
                        for i, (name, count, score) in enumerate(top_metabolites, 1):
                            score_str = f", score={score:.3f}" if score is not None else ""
                            print(f"     {i}. {name} (n={count}{score_str})")
                    else:
                        top_metabolites = []
                        print(f"\n   Top Known Metabolites: None (all unidentified)")
                    
                    # Note: translated comment removed.
                    text_hits = []
                    avg_sim = None
                    cluster_embeddings = None
                    
                    if cluster_positions:
                        cluster_embeddings = torch.tensor(embeds[cluster_positions])
                    
                    if retrieval_method == 'voting' and cluster_embeddings is not None and len(cluster_embeddings) > 0:
                        # Voting-based retrieval: 每个MS2都检索，然后统计频数
                        try:
                            print(f"\n   Retrieving functional texts using VOTING method...")
                            text_hits = self.retrieve_function_texts_by_voting(
                                cluster_embeddings,
                                top_k_per_sample=top_k,
                                final_top_k=top_k,
                                min_similarity=min_similarity
                            )
                            print(f"     OK Retrieved {len(text_hits)} metabolites by voting")
                            
                            if text_hits:
                                avg_sim = sum(h['max_similarity'] for h in text_hits[:5]) / min(5, len(text_hits))
                                print(f"\n   Top 3 Retrieved by Voting:")
                                for i, hit in enumerate(text_hits[:3], 1):
                                    vote_info = f", votes={hit.get('vote_count', 'N/A')}" if 'vote_count' in hit else ""
                                    print(f"     {i}. {hit['name']} (similarity: {hit['max_similarity']:.4f}{vote_info})")
                                    for j, txt in enumerate(hit['texts'], 1):
                                        preview = txt[:100] + "..." if len(txt) > 100 else txt
                                        print(f"        Text {j}: {preview}")
                                
                                if avg_sim is not None and avg_sim < 0.7:
                                    print(f"\n      WARNING: Low retrieval quality (avg sim={avg_sim:.3f})")
                            
                        except Exception as e:
                            print(f"     OK Voting retrieval failed: {e}, falling back to centroid...")
                            import traceback
                            traceback.print_exc()
                            # Fallback to centroid method
                            if centroid is not None:
                                text_hits = self.retrieve_function_texts_from_centroid(centroid, top_k=top_k)
                    
                    elif centroid is not None:
                        # Centroid-based retrieval (original method)
                        try:
                            print(f"\n   Retrieving functional texts from centroid...")
                            text_hits = self.retrieve_function_texts_from_centroid(centroid, top_k=top_k)
                            print(f"     OK Retrieved {len(text_hits)} unique metabolites with function texts")
                            
                            # Note: translated comment removed.
                            if text_hits:
                                avg_sim = sum(h['max_similarity'] for h in text_hits[:5]) / min(5, len(text_hits))
                                print(f"\n   Top 3 Retrieved Function Texts:")
                                for i, hit in enumerate(text_hits[:3], 1):
                                    print(f"     {i}. {hit['name']} (similarity: {hit['max_similarity']:.4f})")
                                    for j, txt in enumerate(hit['texts'], 1):
                                        preview = txt[:100] + "..." if len(txt) > 100 else txt
                                        print(f"        Text {j}: {preview}")
                                
                                # Note: translated comment removed.
                                if avg_sim is not None and avg_sim < 0.7:
                                    print(f"\n      WARNING: Low retrieval quality (avg sim={avg_sim:.3f})")
                                    print(f"        OK GPT may rely more on metabolite names than functional texts")
                            
                        except Exception as e:
                            print(f"     OK Text retrieval failed: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Note: translated comment removed.
                    
                    if self.gpt:
                        try:
                            theme_prompt = None
                            print(f"\n   Calling GPT to generate functional theme name...")
                            print(f"     Input to GPT:")
                            print(f"       - Cluster stats: {info['known_count']} known, {info['unknown_count']} unknown, {direction_label}")
                            print(f"       - Top metabolites: {len(top_metabolites)} provided")
                            print(f"       - Retrieved texts: {len(text_hits)} hits")
                            if text_hits:
                                avg_sim = sum(h['max_similarity'] for h in text_hits[:5]) / min(5, len(text_hits))
                                print(f"       - Avg similarity: {avg_sim:.3f} {' LOW' if avg_sim < 0.7 else 'OK'}")
                            print(f"       - Background info: {'Provided' if background_info else 'Not provided'}")
                            
                            # Note: translated comment removed.
                            functional_name, theme_prompt = self.gpt.generate_cluster_functional_name(
                                cluster_stats=info,
                                top_metabolites=top_metabolites,
                                retrieved_texts=text_hits,
                                background_info=background_info,
                                debug_output_path=None,
                                return_prompt=True
                            )
                            print(f"     OK Generated theme: '{functional_name}'")
                        except Exception as e:
                            print(f"     OK GPT theme name failed: {e}")
                            functional_name = f"Metabolic Cluster {cid+1}"
                            theme_prompt = None
                    else:
                        print(f"\n   GPT disabled - using default name")
                        functional_name = f"Metabolic Cluster {cid+1}"
                        theme_prompt = None
                    
                    # Note: translated comment removed.
                    cluster_report = ""
                    if self.gpt:
                        try:
                            print(f"\n   Calling GPT to generate cluster report...")
                            cluster_report = self.gpt.generate_cluster_report(
                                cluster_stats=info,
                                functional_name=functional_name,
                                top_metabolites=top_metabolites,
                                retrieved_texts=text_hits,
                                example_logfc=c_df[col_fc].tolist() if col_fc else None,
                                background_info=background_info
                            )
                            preview = cluster_report[:150] + "..." if len(cluster_report) > 150 else cluster_report
                            print(f"     OK Report generated ({len(cluster_report)} chars)")
                            print(f"     Preview: {preview}")
                        except Exception as e:
                            print(f"     OK GPT report failed: {e}")
                            cluster_report = f"Report generation failed for {functional_name}."
                    else:
                        cluster_report = f"GPT is disabled. Enable it by setting OPENAI_API_KEY."
                    
                    # Note: translated comment removed.
                    papers = []
                    if len(c_df) > 2:  # cluster
                        try:
                            if self.gpt:
                                # Note: translated comment removed.
                                print(f"\n   Generating intelligent PubMed query...")
                                pubmed_query = self.gpt.generate_pubmed_query(
                                    functional_name=functional_name,
                                    top_metabolites=top_metabolites,
                                    background_info=background_info
                                )
                                print(f"     OK Query: '{pubmed_query}'")
                            else:
                                # Note: translated comment removed.
                                pubmed_query = top_metabolites[0][0] if top_metabolites else functional_name
                                print(f"\n   Searching PubMed (fallback mode): '{pubmed_query}'...")
                            
                            papers = self.pubmed.search_by_metabolites([pubmed_query], max_results=3)
                            print(f"     OK Found {len(papers)} papers")
                            if papers:
                                for i, p in enumerate(papers, 1):
                                    print(f"       {i}. {p['title'][:60]}... (PMID: {p['pmid']})")
                        except Exception as e:
                            print(f"      PubMed search failed: {e}")
                    else:
                        print(f"\n   Skipping PubMed search (cluster too small: {len(c_df)} samples)")
                    
                    # Note: translated comment removed.
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
                        "top_metabolites": [t[0] for t in top_metabolites],
                        "papers": papers,
                        "gpt_theme_prompt": theme_prompt,
                        "retrieved_metabolites": [
                            {
                                "name": h.get("name", ""),
                                "similarity": h.get("max_similarity", 0),
                                "vote_count": h.get("vote_count"),
                                "texts": h.get("texts", [])
                            }
                            for h in text_hits
                        ] if text_hits else []
                    }
                    
                    cluster_reports.append(cluster_info)
                    print(f"\n  OK Cluster {cid+1} complete: '{functional_name}'")
                    print("=" * 80)
                    
                except Exception as e:
                    print(f"OK Failed to process cluster {cid+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
            print(f"OK Generated {len(cluster_reports)} cluster reports")
            
            # 原代码：
            cluster_reports.sort(key=lambda x: abs(x['avg_logfc']), reverse=True)

            # 【新增】后处理：去重复theme name
            if self.gpt:
                print("\n Disambiguating duplicate theme names...")
                cluster_reports = self.gpt.disambiguate_duplicate_themes(cluster_reports)

            # Note: translated comment removed.
            print(" Generating global biological narrative with GPT (Tier 2)...")

            print("=" * 80)
            try:
                if self.gpt:
                    print(f"\n Input Summary for Global Story:")
                    print(f"   - Number of clusters: {len(cluster_reports)}")
                    print(f"   - Background info provided: {'Yes' if background_info else 'No'}")
                    print(f"\n   Cluster Themes:")
                    for i, cr in enumerate(cluster_reports, 1):
                        icon = "" if "Up" in cr['direction'] else ""
                        print(f"     {i}. {icon} {cr['functional_name']} (logFC: {cr['avg_logfc']:.2f})")
                    
                    print(f"\n Calling GPT to synthesize global narrative...")
                    story = self.gpt.generate_global_story(
                        cluster_reports=cluster_reports,
                        background_info=background_info
                    )
                    print(f"OK Global story generated ({len(story)} chars)")
                    print(f"\n Story Preview (first 300 chars):")
                    print(f"   {story[:300]}...")
                else:
                    story = "GPT is disabled. Enable it by setting OPENAI_API_KEY."
                    print(" GPT disabled - no story generated")
                print("=" * 80)
            except Exception as e:
                print(f"OK GPT global story generation failed: {e}")
                import traceback
                traceback.print_exc()
                story = f"GPT generation failed: {str(e)}"
            
            # Note: translated comment removed.
            plot_df = df.loc[valid_idx].copy()
            if col_anno:
                plot_df['Annotator'] = plot_df[col_anno]
            else:
                plot_df['Annotator'] = 'NA'
            
            # Note: translated comment removed.
            if 'variable_id' not in plot_df.columns:
                plot_df['variable_id'] = plot_df.index.astype(str)
            
            result = {
                "story": story,
                "clusters": cluster_reports,
                "plot_data": plot_df[['variable_id', 'tsne_x', 'tsne_y', 'cluster_id', 'Annotator']].to_dict('records'),
            }
            
            print(f"OK Analysis complete! {len(cluster_reports)} clusters, {len(plot_df)} samples")
            return result
            
        except Exception as e:
            print(f"OK Fatal error in run_semi_supervised_analysis: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed: {str(e)}"}

manager = Spec2FunctionManager()