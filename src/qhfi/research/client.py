"""Clients for the local LLM stack.

The framework is a *client* of your existing services — it does not embed LangGraph/crewAI:

  * ``LLMClient``       → OpenAI-compatible vLLM auto-swap proxy (:8001) for single-shot
                          completions / structured output (ideation, critique).
  * ``LangGraphBridge`` → POST to the LangGraph FastAPI service (:8082) for multi-step graphs.
  * ``CrewAIBridge``    → POST to the crewAI FastAPI service (:8083) for crew decomposition.

All use httpx. Structured output is requested via the proxy's OpenAI-compatible
``response_format`` (json schema) so agents return validated objects, not prose.
"""

from __future__ import annotations

import json
from typing import Any

from qhfi.api.client import ManagedHttpClient
from qhfi.api.registry import ApiRegistry
from qhfi.core.config import Settings, get_settings


class LLMClient:
    """Calls the OpenAI-compatible vLLM proxy through the managed API layer (rate-limit,
    retry/backoff, circuit breaker). Inject ``http`` (a ManagedHttpClient over an httpx
    MockTransport) in tests to stay offline."""

    def __init__(self, settings: Settings | None = None, http: ManagedHttpClient | None = None) -> None:
        self.s = settings or get_settings()
        self.http = http or ApiRegistry(self.s).client("llm")

    def complete(self, system: str, user: str, model: str | None = None, **kw: Any) -> str:
        """Single chat completion against the proxy → assistant text.

        Note: per memory, Qwen3.6 reasoning is default-off at the proxy and gemma needs its
        reasoning/tool parsers paired — model choice is the proxy's concern, not ours.
        """
        payload = {
            "model": model or self.s.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kw,
        }
        data = self.http.post("/chat/completions", json=payload)
        return data["choices"][0]["message"]["content"]

    def structured(self, system: str, user: str, schema: dict, model: str | None = None) -> dict:
        """Completion constrained to a JSON schema via response_format → parsed dict."""
        payload = {
            "model": model or self.s.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
        }
        data = self.http.post("/chat/completions", json=payload)
        return json.loads(data["choices"][0]["message"]["content"])


class LangGraphBridge:
    def __init__(self, settings: Settings | None = None) -> None:
        self.url = (settings or get_settings()).langgraph_url

    def invoke(self, graph: str, payload: dict) -> dict:
        raise NotImplementedError("TODO: POST {url}/invoke/{graph} → result")


class CrewAIBridge:
    def __init__(self, settings: Settings | None = None) -> None:
        self.url = (settings or get_settings()).crewai_url

    def kickoff(self, crew: str, inputs: dict) -> dict:
        raise NotImplementedError("TODO: POST {url}/kickoff/{crew} → result")
