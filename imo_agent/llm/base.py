"""Provider-neutral model interface.

The canonical shape keeps the system prompt *separate* from the message list,
because that is what the underlying APIs want (Anthropic `system=`, Gemini
`systemInstruction`).  Flattening it into the messages loses that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..state import Message


@dataclass
class ModelConfig:
    name: str
    api: str
    model: str
    base_url: str | None = None
    key_env: str | None = None
    max_tokens: int = 32000
    effort: str | None = None
    temperature: float | None = None
    thinking: str = "adaptive"
    # Compat servers disagree on this: OpenAI wants max_completion_tokens,
    # vLLM/Ollama/Gemini-compat want max_tokens.
    max_tokens_param: str = "max_completion_tokens"
    fallbacks: bool = False
    strip_preamble: bool = False
    timeout: float = 3600.0
    max_retries: int = 5
    extra: dict = field(default_factory=dict)


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class RefusalError(RuntimeError):
    """The model declined the request (Anthropic stop_reason == 'refusal')."""


class AuthError(RuntimeError):
    """No usable credentials. Fatal - retrying the run will not help."""


class LLMClient(Protocol):
    cfg: ModelConfig

    def complete(self, system: str, messages: list[Message]) -> Completion: ...


def build_client(cfg: ModelConfig) -> LLMClient:
    if cfg.api == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(cfg)
    if cfg.api == "openai_compat":
        from .openai_compat import OpenAICompatClient

        return OpenAICompatClient(cfg)
    if cfg.api == "claude_cli":
        from .claude_cli import ClaudeCLIClient

        return ClaudeCLIClient(cfg)
    raise ValueError(
        f"unknown api {cfg.api!r} for model {cfg.name!r} "
        f"(expected 'anthropic', 'openai_compat' or 'claude_cli')"
    )
