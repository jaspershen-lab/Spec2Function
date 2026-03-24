from __future__ import annotations

import json
import urllib.request
from typing import List, Dict, Optional

from openai import OpenAI


class LLMClient:
    """Provider-agnostic chat completion client."""

    def __init__(self, provider: str, api_key: str, base_url: Optional[str] = None):
        self.provider = (provider or "openai").lower()
        self.api_key = api_key or ""
        self.base_url = (base_url or "").strip()
        normalized_base_url = self._normalize_openai_base_url(self.base_url)

        if self.provider in {"openai", "siliconflow"}:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=normalized_base_url or None
            )
        else:
            self.client = None

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        model: str
    ) -> str:
        if self.provider in {"openai", "siliconflow"}:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

        if self.provider == "gemini":
            return self._gemini_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model
            )

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    @staticmethod
    def _normalize_openai_base_url(base_url: str) -> str:
        if not base_url:
            return ""
        url = base_url.rstrip("/")
        suffix = "/chat/completions"
        if url.endswith(suffix):
            url = url[:-len(suffix)]
        return url

    def _gemini_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        model: str
    ) -> str:
        if not self.api_key:
            raise ValueError("Gemini API key is missing.")

        base_url = self.base_url.rstrip("/") if self.base_url else "https://generativelanguage.googleapis.com/v1beta/models"
        url = f"{base_url}/{model}:generateContent?key={self.api_key}"

        system_message = next((m["content"] for m in messages if m.get("role") == "system"), "")
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                continue
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": msg.get("content", "")}]
            })

        payload = {
            "contents": contents or [{"role": "user", "parts": [{"text": ""}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens)
            }
        }
        if system_message:
            payload["systemInstruction"] = {"parts": [{"text": system_message}]}

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        candidates = response_data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return ""
        return (parts[0].get("text") or "").strip()
