import pytest

from imo_agent.expr import ExprError, evaluate, identifiers


def test_comparison_and_boolean_ops():
    ns = {"streak": 5, "errors": 0, "iterations": 3}
    assert evaluate("streak >= 5", ns)
    assert not evaluate("errors >= 10 or iterations >= 30", ns)
    assert evaluate("streak > 1 and errors == 0", ns)


def test_string_equality_for_labels():
    assert evaluate("verdict_label == 'no'", {"verdict_label": "no"})


def test_unknown_name_is_reported():
    with pytest.raises(ExprError, match="unknown name 'typo'"):
        evaluate("typo >= 1", {"streak": 1})


def test_calls_are_rejected():
    with pytest.raises(ExprError):
        evaluate("__import__('os').system('true')", {})


def test_identifiers_are_extracted():
    assert identifiers("errors >= 10 or iterations >= 30") == {"errors", "iterations"}
