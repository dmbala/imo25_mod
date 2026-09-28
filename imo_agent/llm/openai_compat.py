"""OpenAI-compatible chat/completions adapter.

Covers OpenAI itself plus every server that speaks the same wire format:
xAI, DeepSeek, OpenRouter, Gemini's compat endpoint, and local vLLM/Ollama.
"""

from __future__ import annotations

import os

import openai

from ..state import Message
from .base import Completion, ModelConfig


class OpenAICompatClient:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        key = os.environ.get(cfg.key_env) if cfg.key_env else None
        if cfg.key_env and not key and not _is_local(cfg.base_url):
            raise RuntimeError(
                f"{cfg.key_env} is not set (needed by model '{cfg.name}'). "
                f"Export it, e.g. export {cfg.key_env}=..."
            )
        self._client = openai.OpenAI(
            # Local servers usually run unauthenticated but the SDK insists on a value.
            api_key=key or "EMPTY",
            base_url=cfg.base_url,
            max_retries=cfg.max_retries,
            timeout=cfg.timeout,
        )

    def complete(self, system: str, messages: list[Message]) -> Completion:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(m.as_dict() for m in messages)

        params: dict = {"model": self.cfg.model, "messages": msgs}
        if self.cfg.max_tokens:
            params[self.cfg.max_tokens_param] = self.cfg.max_tokens
        if self.cfg.temperature is not None:
            params["temperature"] = self.cfg.temperature
        # Non-reasoning models and many compat servers 400 on this.
        if self.cfg.effort:
            params["reasoning_effort"] = self.cfg.effort
        params.update(self.cfg.extra)

        resp = self._client.chat.completions.create(**params)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return Completion(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=getattr(resp, "model", self.cfg.model),
        )


def _is_local(base_url: str | None) -> bool:
    if not base_url:
        return False
    return any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
