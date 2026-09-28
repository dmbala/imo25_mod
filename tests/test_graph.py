"""Full walks of the shipped pipeline against a scripted stub model."""

from __future__ import annotations

from imo_agent import graph as graph_mod
from imo_agent.state import RunState


def walk(ctx):
    state = RunState(problem="Prove that 1 + 1 = 2.", other_prompts=[])
    return graph_mod.run(state, ctx.cfg, ctx)


def test_accepts_after_five_consecutive_passes(make_ctx):
    # streak starts at 1, so four more passes reach the accept threshold.
    ctx, client = make_ctx(verdicts=["yes"] * 4)
    outcome = walk(ctx)

    assert outcome.status == "accept"
    assert outcome.state.counters["streak"] == 5
    assert outcome.state.counters["errors"] == 0
    assert "Detailed Solution" in outcome.solution


def test_fails_once_errors_reach_ten(make_ctx):
    # errors is bumped entering a correction and tested after the next verify,
    # so the eleventh rejection is the one that ends the run.
    ctx, client = make_ctx(verdicts=["no"] * 12)
    outcome = walk(ctx)

    assert outcome.status == "fail"
    assert outcome.state.counters["errors"] == 10


def test_rejection_feeds_the_bug_report_into_the_correction(make_ctx):
    # A rejection zeroes the streak, so five passes are needed afterwards.
    ctx, client = make_ctx(verdicts=["no"] + ["yes"] * 5)
    outcome = walk(ctx)

    assert outcome.status == "accept"
    assert outcome.state.get("bug_report")
    corrections = [
        payload for _system, payload in client.calls
        if any("bug report" in m["content"] for m in payload)
    ]
    assert corrections, "no correction turn was ever sent"
    last_user = corrections[0][-1]["content"]
    assert "Final Verdict" in last_user      # the bug report body was appended
    assert any(m["role"] == "assistant" for m in corrections[0])


def test_streak_resets_on_a_rejection(make_ctx):
    # Two passes, a rejection, then five more. Without the reset the run would
    # have accepted after the fourth pass; with it, all eight judgements run.
    ctx, client = make_ctx(verdicts=["yes", "yes", "no"] + ["yes"] * 5)
    outcome = graph_mod.run(RunState(problem="p"), ctx.cfg, ctx)

    judgements = [
        payload for _system, payload in client.calls
        if 'Response in "yes" or "no"' in payload[-1]["content"]
    ]
    assert outcome.status == "accept"
    assert len(judgements) == 8
    assert outcome.state.get("bug_report")          # the rejection did trigger a correction
    assert outcome.state.counters["errors"] == 0    # a later pass clears the error count


def test_max_steps_is_enforced(make_ctx):
    ctx, client = make_ctx(verdicts=["no"] * 500)
    ctx.cfg.pipeline.max_steps = 6
    outcome = graph_mod.run(RunState(problem="p"), ctx.cfg, ctx)
    assert outcome.status == "exhausted"


def test_mermaid_export_covers_every_node(cfg):
    diagram = graph_mod.to_mermaid(cfg)
    assert diagram.startswith("flowchart TD")
    for name in cfg.pipeline.nodes:
        assert name in diagram
    assert "accept" in diagram and "fail" in diagram
