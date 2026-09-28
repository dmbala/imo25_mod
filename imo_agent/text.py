"""Text surgery lifted from the original agents, kept as pure functions."""

from __future__ import annotations


def split_marker(text: str, marker: str, keep: str = "after") -> str:
    """Port of extract_detailed_solution(): '' when the marker is absent."""
    idx = text.find(marker)
    if idx == -1:
        return ""
    if keep == "after":
        return text[idx + len(marker):].strip()
    return text[:idx].strip()


def strip_preamble(text: str) -> str:
    """Port of agent_xai.extract_solution(): drop chatter before the last Summary."""
    idx = text.rfind("Summary")
    if idx == -1:
        return text
    if "### " in text[max(0, idx - 4):idx]:
        return text[idx - 4:].strip()
    return text[idx:].strip()
