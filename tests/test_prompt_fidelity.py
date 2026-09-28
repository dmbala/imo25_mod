"""The prompts must stay byte-identical to the originals they were lifted from.

Skipped when the upstream checkout is not beside this repo.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT.parent / "IMO25" / "code" / "agent.py"

MAPPING = {
    "step1_prompt": "solve_system.md",
    "self_improvement_prompt": "self_improve.md",
    "check_verification_prompt": "check_verification.md",
    "correction_prompt": "correction.md",
    "verification_system_prompt": "verify_system.md",
    "verification_remider": "verify_reminder.md",
}


def _upstream_strings() -> dict[str, str]:
    tree = ast.parse(UPSTREAM.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in MAPPING:
                    out[target.id] = node.value.value
    return out


@pytest.mark.skipif(not UPSTREAM.exists(), reason="upstream IMO25 checkout not present")
@pytest.mark.parametrize("name,filename", sorted(MAPPING.items()))
def test_prompt_matches_upstream(name, filename):
    expected = _upstream_strings()[name]
    actual = (ROOT / "prompts" / filename).read_text(encoding="utf-8")
    assert actual == expected
