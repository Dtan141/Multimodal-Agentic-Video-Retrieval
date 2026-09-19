"""OpenAI-compatible LLM client (Groq free tier).

Thin httpx wrapper over ``/chat/completions`` with JSON-mode + retry/backoff for
the free-tier rate limits. Base URL / model / key come from Settings.
"""

from __future__ import annotations

import json
import time

import httpx

from aic_retrieval.config import Settings, get_settings
from aic_retrieval.logging_conf import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LLMClient:
        settings = settings or get_settings()
        return cls(settings.groq_base_url, settings.groq_model, settings.groq_api_key or "", settings.llm_timeout)

    def _post(self, payload: dict, *, max_retries: int = 3) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("retry-after", delay))
                    logger.warning("Groq 429, retry in %.1fs (attempt %d)", wait, attempt + 1)
                    time.sleep(wait)
                    delay *= 2
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_exc}")

    def chat_json(self, system: str, user: str, *, temperature: float = 0.0) -> dict:
        """Chat expecting a JSON object back. Robust to code-fenced output."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        data = self._post(payload)
        content = data["choices"][0]["message"]["content"]
        return _parse_json(content)


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # strip ```json fences or extract the first {...} block
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
