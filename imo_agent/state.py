"""Run state threaded through every node of the pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Message:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class RunState:
    problem: str = ""
    other_prompts: list[str] = field(default_factory=list)
    system: str = ""
    conversation: list[Message] = field(default_factory=list)
    # Named slots written by `into:` and read back by {placeholders} in templates.
    fields: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    current_node: str = ""
    step: int = 0

    def get(self, name: str, default: str = "") -> str:
        if name == "problem":
            return self.problem
        return self.fields.get(name, default)

    def set(self, name: str, value: str) -> None:
        self.fields[name] = value

    def namespace(self) -> dict[str, Any]:
        """Names visible to guard expressions: counters plus string slots."""
        ns: dict[str, Any] = dict(self.counters)
        ns.update(self.fields)
        return ns

    def to_json(self) -> str:
        d = asdict(self)
        d["conversation"] = [m.as_dict() for m in self.conversation]
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "RunState":
        d = json.loads(text)
        d["conversation"] = [Message(**m) for m in d.get("conversation", [])]
        return cls(**d)
