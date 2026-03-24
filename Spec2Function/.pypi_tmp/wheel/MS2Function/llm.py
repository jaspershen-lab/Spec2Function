from __future__ import annotations

from dataclasses import dataclass
import json
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str = "Qwen/Qwen2.5-72B-Instruct"
    base_url: str = "https://api.siliconflow.cn/v1"
    temperature: float = 0.2
    max_tokens: int = 800

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("MS2BIOTEXT_LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing LLM API key: set SILICONFLOW_API_KEY (default) or MS2BIOTEXT_LLM_API_KEY."
            )

        base_url = (
            os.getenv("SILICONFLOW_BASE_URL")
            or os.getenv("MS2BIOTEXT_LLM_BASE_URL")
            or cls.base_url
        )
        model = os.getenv("MS2BIOTEXT_LLM_MODEL") or cls.model

        temperature = float(os.getenv("MS2BIOTEXT_LLM_TEMPERATURE") or cls.temperature)
        max_tokens = int(os.getenv("MS2BIOTEXT_LLM_MAX_TOKENS") or cls.max_tokens)

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url.rstrip("/"),
            temperature=temperature,
            max_tokens=max_tokens,
        )


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat_completions(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            method="POST",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        try:
            obj = json.loads(raw)
            return (obj["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:
            raise RuntimeError(f"LLM response parse failed: {exc}. Raw: {raw[:500]}") from exc


class GPTInference:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        base_url: str = "https://api.siliconflow.cn/v1",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> None:
        self.client = OpenAICompatibleClient(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_env(cls) -> "GPTInference":
        cfg = LLMConfig.from_env()
        return cls(
            api_key=cfg.api_key,
            model=cfg.model,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    def _chat(self, system_prompt: str, user_prompt: str, *, max_tokens: Optional[int] = None) -> str:
        return self.client.chat_completions(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )

    def generate_cluster_functional_name(
        self,
        *,
        cluster_stats: Any,
        top_metabolites: Any,
        retrieved_texts: Any,
        background_info: Optional[str] = None,
        debug_output_path: Optional[str] = None,
        return_prompt: bool = False,
    ) -> Any:
        system_prompt = "You name metabolite clusters based on brief stats and metabolite hints."
        user_prompt = (
            f"Background: {background_info}\n"
            f"Cluster stats: {cluster_stats}\n"
            f"Top metabolites: {top_metabolites}\n"
            f"Retrieved texts: {retrieved_texts}\n"
            "Return a short functional name (<=8 words)."
        )
        name = self._chat(system_prompt, user_prompt, max_tokens=64)
        if debug_output_path:
            try:
                with open(debug_output_path, "w", encoding="utf-8") as f:
                    f.write(user_prompt)
            except Exception:
                pass
        if return_prompt:
            return name, user_prompt
        return name

    def generate_cluster_report(
        self,
        *,
        cluster_stats: Any,
        functional_name: str,
        top_metabolites: Any,
        retrieved_texts: Any,
        example_logfc: Any = None,
        background_info: Optional[str] = None,
    ) -> str:
        system_prompt = "You write concise metabolic cluster reports for biologists."
        user_prompt = (
            f"Background: {background_info}\n"
            f"Cluster name: {functional_name}\n"
            f"Cluster stats: {cluster_stats}\n"
            f"Top metabolites: {top_metabolites}\n"
            f"Example logFC: {example_logfc}\n"
            f"Retrieved texts: {retrieved_texts}\n"
            "Write a 3-6 sentence summary."
        )
        return self._chat(system_prompt, user_prompt, max_tokens=350)

    def generate_pubmed_query(
        self, *, functional_name: str, top_metabolites: Any, background_info: Optional[str] = None
    ) -> str:
        system_prompt = "You generate concise PubMed search queries."
        user_prompt = (
            f"Background: {background_info}\n"
            f"Functional name: {functional_name}\n"
            f"Top metabolites: {top_metabolites}\n"
            "Return a short query string."
        )
        return self._chat(system_prompt, user_prompt, max_tokens=64)

    def generate_global_story(self, *, cluster_reports: Any, background_info: Optional[str] = None) -> str:
        system_prompt = "You summarize multiple metabolic clusters into a short story."
        user_prompt = (
            f"Background: {background_info}\n"
            f"Clusters: {cluster_reports}\n"
            "Write a concise 5-8 sentence overview."
        )
        return self._chat(system_prompt, user_prompt, max_tokens=500)

    def single_annotation(
        self,
        *,
        retrieved_fragments: List[Dict],
        papers: Optional[List[Dict]] = None,
        user_focus: Optional[str] = None,
    ) -> str:
        system_prompt = "You summarize fragment matches into an annotation."
        user_prompt = (
            f"Retrieved fragments: {retrieved_fragments}\n"
            f"Papers: {papers or []}\n"
            f"User focus: {user_focus}\n"
            "Write a short annotation."
        )
        return self._chat(system_prompt, user_prompt, max_tokens=200)

