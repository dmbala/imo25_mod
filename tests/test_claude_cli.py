"""Adapter for the `claude` CLI, exercised against a fake binary (no network)."""

from __future__ import annotations

import json
import os
import stat

import pytest

from imo_agent.llm.base import AuthError, ModelConfig
from imo_agent.llm.claude_cli import ClaudeCLIClient, flatten
from imo_agent.state import Message

SUCCESS = {
    "type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
    "result": "the answer", "stop_reason": "end_turn",
    "usage": {"input_tokens": 2, "output_tokens": 4,
              "cache_creation_input_tokens": 100, "cache_read_input_tokens": 50},
}


def fake_claude(tmp_path, body: str, exit_code: int = 0):
    """A stand-in `claude` that records its argv and prints a canned payload."""
    path = tmp_path / "claude"
    path.write_text(
        "#!/bin/sh\n"
        f'printf "%s" "$*" > "{tmp_path}/argv.txt"\n'
        f'cat > "{tmp_path}/stdin.txt"\n'
        f"cat <<'EOF'\n{body}\nEOF\n"
        f"exit {exit_code}\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def client_for(tmp_path, body=None, exit_code=0, **cfg_kwargs):
    fake_claude(tmp_path, body if body is not None else json.dumps(SUCCESS), exit_code)
    os.environ["PATH"] = f"{tmp_path}{os.pathsep}{os.environ['PATH']}"
    cfg = ModelConfig(name="claude-code", api="claude_cli", model="opus", **cfg_kwargs)
    return ClaudeCLIClient(cfg), tmp_path


def test_single_user_turn_is_passed_through_unlabelled():
    assert flatten([Message("user", "just this")]) == "just this"


def test_multi_turn_is_rendered_as_a_transcript():
    out = flatten([Message("user", "a"), Message("assistant", "b"), Message("user", "c")])
    assert out == "## User\n\na\n\n## Assistant\n\nb\n\n## User\n\nc"


def test_result_and_usage_are_extracted(tmp_path):
    client, work = client_for(tmp_path)
    got = client.complete("SYS", [Message("user", "hello")])

    assert got.text == "the answer"
    assert got.output_tokens == 4
    assert got.input_tokens == 152          # 2 + 100 cache-create + 50 cache-read
    assert (work / "stdin.txt").read_text() == "hello"


def test_invocation_strips_tools_and_passes_the_system_prompt(tmp_path):
    client, work = client_for(tmp_path, effort="high")
    client.complete("SYSTEM TEXT", [Message("user", "hi")])
    argv = (work / "argv.txt").read_text()

    assert "--print" in argv and "--output-format json" in argv
    assert "--restricted" in argv                     # no Bash/REPL tools
    assert "--permission-prompts none" in argv        # nothing can prompt
    assert "--strict-mcp-config" in argv
    assert "--system-prompt SYSTEM TEXT" in argv
    assert "--model opus" in argv
    assert "--effort high" in argv


def test_error_payload_is_raised(tmp_path):
    body = json.dumps({"is_error": True, "subtype": "error_during_execution",
                       "result": None, "api_error_status": "overloaded"})
    client, _ = client_for(tmp_path, body=body)
    with pytest.raises(RuntimeError, match="overloaded"):
        client.complete("", [Message("user", "hi")])


def test_nonzero_exit_mentioning_login_becomes_an_auth_error(tmp_path):
    client, _ = client_for(tmp_path, body="Please run claude login first", exit_code=1)
    with pytest.raises(AuthError, match="not logged in"):
        client.complete("", [Message("user", "hi")])


def test_banner_before_the_json_is_tolerated(tmp_path):
    client, _ = client_for(tmp_path, body="warning: update available\n" + json.dumps(SUCCESS))
    assert client.complete("", [Message("user", "hi")]).text == "the answer"


def test_unparseable_output_is_reported(tmp_path):
    client, _ = client_for(tmp_path, body="not json at all")
    with pytest.raises(RuntimeError, match="did not return JSON"):
        client.complete("", [Message("user", "hi")])


def test_missing_binary_is_reported():
    cfg = ModelConfig(name="x", api="claude_cli", model="opus",
                      extra={"binary": "definitely-not-a-real-binary"})
    with pytest.raises(RuntimeError, match="on PATH"):
        ClaudeCLIClient(cfg)
