"""Placeholder substitution.

Uses an explicit regex rather than str.format because every prompt in this
project is full of TeX -- `\\frac{a}{b}`, `$x_{i}$` -- and str.format would
choke on those braces or silently eat them.  Unknown placeholders are left
untouched for the same reason.
"""

from __future__ import annotations

import re

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render(template: str, values: dict[str, str]) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key in values and values[key] is not None:
            return str(values[key])
        return match.group(0)

    return _PLACEHOLDER.sub(repl, template)


def placeholders(template: str) -> set[str]:
    return set(_PLACEHOLDER.findall(template))
