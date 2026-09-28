from imo_agent.text import split_marker, strip_preamble


def test_split_after_returns_tail():
    assert split_marker("intro Detailed Solution  body", "Detailed Solution") == "body"


def test_split_before_returns_head():
    assert split_marker("head Detailed Verification tail", "Detailed Verification", "before") == "head"


def test_missing_marker_returns_empty():
    assert split_marker("nothing here", "Detailed Solution") == ""


def test_strip_preamble_keeps_heading():
    assert strip_preamble("chatter\n### Summary\nreal").startswith("### Summary")


def test_strip_preamble_without_summary_is_identity():
    assert strip_preamble("no marker at all") == "no marker at all"
