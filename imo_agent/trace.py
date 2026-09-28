"""Human-readable log plus a structured JSONL record per node execution."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


class Tracer:
    def __init__(self, log_path: str | None = None, trace_path: str | None = None, echo: bool = True):
        self.echo = echo
        self._log = open(log_path, "w", encoding="utf-8") if log_path else None
        self._trace = open(trace_path, "w", encoding="utf-8") if trace_path else None

    def log(self, message: str) -> None:
        message = str(message)
        if message.startswith(">>>>>"):
            message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        if self.echo:
            print(message, file=sys.stdout, flush=True)
        if self._log:
            self._log.write(message + "\n")
            self._log.flush()

    def record(self, payload: dict) -> None:
        if not self._trace:
            return
        payload = {"ts": datetime.now().isoformat(), **payload}
        self._trace.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._trace.flush()

    def close(self) -> None:
        for handle in (self._log, self._trace):
            if handle:
                handle.close()
        self._log = self._trace = None

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def save_checkpoint(path: str | Path, state) -> None:
    Path(path).write_text(state.to_json(), encoding="utf-8")


def load_checkpoint(path: str | Path):
    from .state import RunState

    return RunState.from_json(Path(path).read_text(encoding="utf-8"))
