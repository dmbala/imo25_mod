"""Anthropic Messages API adapter."""

from __future__ import annotations

import os

import anthropic

from ..state import Message
from .base import AuthError, Completion, ModelConfig, RefusalError

AUTH_HELP = """no usable Anthropic credentials found. Any one of these works:

  1. An OAuth profile, no static key to manage (reads ~/.config/anthropic/):
       ant auth login                  # --no-browser on a headless node
  2. Feed this shell the active profile's short-lived token:
       set -a; eval "$(ant auth print-credentials --env)"; set +a
  3. An API key from console.anthropic.com:
       export ANTHROPIC_API_KEY=...

Note: Claude Code's own login in ~/.claude/.credentials.json is a separate
store that the anthropic SDK does not read."""


class AnthropicClient:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        kwargs: dict = {"max_retries": cfg.max_retries, "timeout": cfg.timeout}
        # Pass a key only when there is a real one. An empty string still wins
        # the SDK's precedence slot and authenticates as an empty key, which
        # would shadow an `ant auth login` profile or ANTHROPIC_AUTH_TOKEN.
        key = os.environ.get(cfg.key_env) if cfg.key_env else None
        if key:
            kwargs["api_key"] = key
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self._client = anthropic.Anthropic(**kwargs)
        self._fallbacks = cfg.fallbacks

    def complete(self, system: str, messages: list[Message]) -> Completion:
        params: dict = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "messages": [m.as_dict() for m in messages],
        }
        if system:
            params["system"] = system
        if self.cfg.thinking and self.cfg.thinking != "off":
            params["thinking"] = {"type": self.cfg.thinking}
        if self.cfg.effort:
            params["output_config"] = {"effort": self.cfg.effort}
        # Current Claude models reject temperature; only send when explicitly configured.
        if self.cfg.temperature is not None:
            params["temperature"] = self.cfg.temperature
        params.update(self.cfg.extra)

        try:
            msg = self._send(params)
        except anthropic.AuthenticationError as exc:
            raise AuthError(f"{AUTH_HELP}\n\n(server said: {exc})") from exc
        except TypeError as exc:
            # The SDK raises this before sending when it can resolve no credential.
            if "authentication" not in str(exc).lower():
                raise
            raise AuthError(AUTH_HELP) from exc

        if msg.stop_reason == "refusal":
            category = getattr(msg.stop_details, "category", None)
            raise RefusalError(f"model declined the request (category={category!r})")

        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = msg.usage
        return Completion(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=getattr(msg, "model", self.cfg.model),
        )

    def _send(self, params: dict):
        """Streaming, not create(): max_tokens is large and a hard proof can
        generate for many minutes, which is exactly what trips the HTTP timeout."""
        if self._fallbacks:
            try:
                return self._stream(
                    self._client.beta.messages,
                    dict(params, betas=["server-side-fallback-2026-07-01"], fallbacks="default"),
                )
            except Exception as exc:
                if not _is_unsupported_parameter(exc):
                    raise
                # Older SDK or an endpoint without the beta: drop it and carry on.
                self._fallbacks = False
        return self._stream(self._client.messages, params)

    @staticmethod
    def _stream(endpoint, params: dict):
        with endpoint.stream(**params) as stream:
            return stream.get_final_message()


def _is_unsupported_parameter(exc: Exception) -> bool:
    if isinstance(exc, TypeError):
        return "fallbacks" in str(exc) or "betas" in str(exc)
    if isinstance(exc, getattr(anthropic, "BadRequestError", ())):
        blob = str(exc).lower()
        return "fallback" in blob or "beta" in blob
    return False
