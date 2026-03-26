# -*- coding: utf-8 -*-
"""
Spec2Function Backend Configuration
"""
from pathlib import Path
import os
from typing import Optional
from .assets import resolve_assets_root
from dotenv import load_dotenv

load_dotenv()

class Spec2FunctionConfig:
    """Global configuration manager."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Args:
            project_root: Project root directory. If None, infer automatically.
        """
        if project_root is None:
            # Infer: parent directory of this file (backend -> root)
            self.project_root = resolve_assets_root(project_root)
        else:
            self.project_root = Path(project_root)

        # Model file paths
        self.model_dir = self.project_root / "models"
        self.model_checkpoint = self.model_dir / "best_model.pth"
        self.model_config = self.model_dir / "config.json"

        # Data file paths
        self.data_dir = self.project_root / "data"
        # Ensure names here match the local files.
        self.femdb_jsonl = self.data_dir / "hmdb_subsections_WITH_NAME.jsonl"
        self.femdb_embeddings = self.data_dir / "all_jsonl_embeddings.pt"

        # Optional alias for compatibility
        self.hmdb_metadata_path = self.femdb_jsonl

        # SPECTER model
        self.specter_model_name = "allenai/specter"
        self.specter_cache_dir = self.model_dir / "specter_cache"

        # Retrieval params
        # self.default_top_k = 10
        self.single_retrieval_top_k = 5      
        self.set_retrieval_top_k = 20          
        self.min_similarity = 0.65

        # GPT params (fixes missing attributes)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.gpt_model = "gpt-4o"  # Or "gpt-4o", "gpt-3.5-turbo"
        self.gpt_max_tokens = 2000  # Mode 2 needs more tokens
        self.gpt_temperature = 0.4  # Previously missing line that caused errors

        # LLM provider params
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "")
        self.llm_model = os.getenv("LLM_MODEL", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY", "")

        if not self.llm_model:
            if self.llm_provider == "siliconflow":
                self.llm_model = "deepseek-ai/DeepSeek-V3.2"
            else:
                self.llm_model = self.gpt_model

        if not self.llm_base_url and self.llm_provider == "siliconflow":
            self.llm_base_url = "https://api.siliconflow.cn/v1/chat/completions"

        # PubMed params
        self.pubmed_email = os.getenv("PUBMED_EMAIL", "your_email@example.com")
        self.pubmed_max_results = 5

        # MSBERT params
        self.msbert_vocab_size = 100002
        self.msbert_hidden_size = 512
        self.msbert_num_layers = 6
        self.msbert_num_heads = 16
        self.msbert_dropout = 0
        self.msbert_max_len = 100
        self.msbert_kernel_size = 3

    def validate(self):
        """Validate required files exist."""
        missing = []

        if not self.model_checkpoint.exists():
            missing.append(f"Model checkpoint: {self.model_checkpoint}")

        if not self.model_config.exists():
            missing.append(f"Model config: {self.model_config}")

        # Temporarily skip strict data file checks if files are not downloaded yet.
        # if not self.femdb_jsonl.exists():
        #     missing.append(f"FemDB JSONL: {self.femdb_jsonl}")

        if missing:
            raise FileNotFoundError(
                "Missing required files:\n" + "\n".join(f"  - {m}" for m in missing)
            )

        if not self.resolve_llm_api_key():
            print("Warning: LLM API key not set. GPT inference will fail.")

        return True

    def resolve_llm_api_key(self) -> str:
        provider = (self.llm_provider or "openai").lower()
        if provider == "gemini":
            return self.gemini_api_key or self.llm_api_key
        if provider == "siliconflow":
            return self.siliconflow_api_key or self.llm_api_key
        return self.openai_api_key or self.llm_api_key


# Global config instance
config = Spec2FunctionConfig()
