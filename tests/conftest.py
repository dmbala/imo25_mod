from __future__ import annotations

from pathlib import Path

import pytest

from imo_agent import config as config_mod
from imo_agent.llm.base import Completion, ModelConfig
from imo_agent.nodes import Context

ROOT = Path(__file__).resolve().parent.parent

SOLUTION = """**1. Summary**
Verdict: complete.

**2. Detailed Solution**
Let $n$ be an integer. The answer is $n = 3$. QED.
"""

VERIFICATION = """**Summary**
Final Verdict: the solution is correct.

**List of Findings**
* none

**Detailed Verification Log**
Step 1 checks out.
"""


class StubClient:
    """Answers by role rather than by call index, so tests survive reordering."""

    def __init__(self, cfg: ModelConfig, verdicts=None, solution=SOLUTION, verification=VERIFICATION):
        self.cfg = cfg
        self.verdicts = list(verdicts or [])
        self.solution = solution
        self.verification = verification
        self.calls: list[tuple[str, list[dict]]] = []

    def complete(self, system, messages):
        payload = [m.as_dict() for m in messages]
        self.calls.append((system, payload))
        last = payload[-1]["content"] if payload else ""

        if 'Response in "yes" or "no"' in last:
            verdict = self.verdicts.pop(0) if self.verdicts else "no"
            return Completion(text=verdict, model="stub")
        if "meticulous grader" in system:
            return Completion(text=self.verification, model="stub")
        return Completion(text=self.solution, model="stub")


@pytest.fixture
def cfg():
    return config_mod.load(ROOT)


@pytest.fixture
def make_ctx(cfg):
    def _make(verdicts=None, **kwargs):
        model = ModelConfig(name="stub", api="anthropic", model="stub")
        client = StubClient(model, verdicts=verdicts, **kwargs)
        ctx = Context(cfg=cfg, clients={"stub": client}, default_model="stub")
        return ctx, client

    return _make
