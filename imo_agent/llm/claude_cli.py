"""Claude Code CLI adapter.

Runs `claude --print` as the model, so the work goes through an existing
Claude Code login instead of an API credential. Claude Code is an agent
harness rather than a completion endpoint, which has consequences:

- Print mode is single-turn, so a multi-turn conversation is flattened into a
  labelled transcript.
- Its own harness context still rides along (~15k cached tokens per call even
  with --system-prompt and --restricted), so this costs noticeably more per
  call than the same model over the API.
- Tools are stripped with --restricted and anything that would prompt is
  denied, because this pipeline only ever needs text in and text out.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile

from ..state import Message
from .base import AuthError, Completion, ModelConfig

ROLE_LABELS = {"user": "## User", "assistant": "## Assistant"}


class ClaudeCLIClient:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.binary = cfg.extra.get("binary", "claude")
        if not shutil.which(self.binary):
            raise RuntimeError(
                f"model '{cfg.name}' needs the '{self.binary}' CLI on PATH. "
                f"Install Claude Code, or set extra.binary to its full path."
            )
        # Run somewhere neutral so CLAUDE.md discovery and repo context do not
        # leak into a maths prompt.
        self._cwd = tempfile.mkdtemp(prefix="imo-claude-cli-")

    def complete(self, system: str, messages: list[Message]) -> Completion:
        cmd = [
            self.binary, "--print",
            "--output-format", "json",
            "--restricted",              # no Bash/REPL/command tools
            "--permission-prompts", "none",
            "--strict-mcp-config",       # ignore the user's MCP servers
        ]
        if self.cfg.model and self.cfg.model != "default":
            cmd += ["--model", self.cfg.model]
        if system:
            cmd += ["--system-prompt", system]
        if self.cfg.effort:
            cmd += ["--effort", self.cfg.effort]
        cmd += self.cfg.extra.get("args", [])

        proc = subprocess.run(
            cmd,
            input=flatten(messages),
            capture_output=True,
            text=True,
            timeout=self.cfg.timeout,
            cwd=self._cwd,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:800]
            if "login" in detail.lower() or "authenticat" in detail.lower():
                raise AuthError(f"`{self.binary}` is not logged in. Run: {self.binary} login\n\n{detail}")
            raise RuntimeError(f"`{self.binary}` exited {proc.returncode}: {detail}")

        payload = _parse(proc.stdout, self.binary)
        if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
            raise RuntimeError(
                f"`{self.binary}` reported {payload.get('subtype')}: "
                f"{payload.get('api_error_status') or payload.get('result')}"
            )

        usage = payload.get("usage") or {}
        return Completion(
            text=payload.get("result") or "",
            # Cached tokens are most of the input here; counting them keeps the
            # trace honest about what this path actually costs.
            input_tokens=(usage.get("input_tokens", 0)
                          + usage.get("cache_creation_input_tokens", 0)
                          + usage.get("cache_read_input_tokens", 0)),
            output_tokens=usage.get("output_tokens", 0),
            model=self.cfg.model or "claude-code",
        )


def flatten(messages: list[Message]) -> str:
    """Print mode takes one prompt, so render the turns as a transcript.

    A lone user message is passed through untouched -- labelling a single turn
    only adds noise to the prompt.
    """
    if len(messages) == 1 and messages[0].role == "user":
        return messages[0].content
    parts = [f"{ROLE_LABELS.get(m.role, '## ' + m.role)}\n\n{m.content}" for m in messages]
    return "\n\n".join(parts)


def _parse(stdout: str, binary: str) -> dict:
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError(f"`{binary}` produced no output")
    try:
        return json.loads(text)
    except ValueError:
        # Be forgiving of a banner or warning printed before the JSON object.
        start = text.find("{")
        if start != -1:
            try:
                return json.loads(text[start:])
            except ValueError:
                pass
        raise RuntimeError(f"`{binary}` did not return JSON: {text[:400]}")
