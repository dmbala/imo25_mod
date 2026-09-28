from imo_agent.config import MessageSpec, NodeConfig
from imo_agent.nodes import apply_counters, node_chat, node_guard, node_transform
from imo_agent.state import Message, RunState


def test_transform_split_writes_slot(cfg, make_ctx):
    ctx, _ = make_ctx()
    state = RunState(problem="p", fields={"solution": "a Detailed Solution b"})
    node = NodeConfig(name="x", type="transform", fn="split", source="solution",
                      marker="Detailed Solution", keep="after", into="detailed")
    node_transform(state, node, ctx)
    assert state.get("detailed") == "b"


def test_guard_accept_and_fail(cfg, make_ctx):
    ctx, _ = make_ctx()
    node = NodeConfig(name="check", type="guard", accept_if="streak >= 5", fail_if="errors >= 10")

    state = RunState(counters={"streak": 5, "errors": 0})
    assert node_guard(state, node, ctx).terminal == "accept"

    state = RunState(counters={"streak": 0, "errors": 10})
    assert node_guard(state, node, ctx).terminal == "fail"

    state = RunState(counters={"streak": 1, "errors": 1})
    assert node_guard(state, node, ctx).terminal is None


def test_guard_when_rule_routes(cfg, make_ctx):
    ctx, _ = make_ctx()
    node = NodeConfig(name="decide", type="guard",
                      when=[{"if": "verdict_label == 'no'", "next": "penalize"}])
    state = RunState(fields={"verdict_label": "no"})
    assert node_guard(state, node, ctx).next == "penalize"


def test_counter_updates_relative_and_absolute():
    state = RunState(counters={"streak": 3, "errors": 2})
    apply_counters(state, {"streak": "+1", "errors": "0"})
    assert state.counters == {"streak": 4, "errors": 0}


def test_chat_reset_rebuilds_conversation(cfg, make_ctx):
    ctx, client = make_ctx()
    node = NodeConfig(
        name="solve", type="chat", reset=True, system="solve_system", into="solution",
        messages=[MessageSpec(role="user", text="{problem}"),
                  MessageSpec(role="user", each="other_prompts")],
    )
    state = RunState(problem="PROBLEM", other_prompts=["use induction"],
                     conversation=[], fields={})
    node_chat(state, node, ctx)

    roles = [m.role for m in state.conversation]
    assert roles == ["user", "user"]
    assert state.conversation[0].content == "PROBLEM"
    assert state.conversation[1].content == "use induction"
    assert "Rigor is Paramount" in client.calls[0][0]   # system prompt reached the client
    assert state.get("solution")


def test_chat_without_reset_appends(cfg, make_ctx):
    ctx, _ = make_ctx()
    node = NodeConfig(name="improve", type="chat", reset=False, into="solution",
                      messages=[MessageSpec(role="assistant", text="{solution}"),
                                MessageSpec(role="user", prompt="self_improve")])
    state = RunState(problem="p", fields={"solution": "first draft"})
    state.conversation = [Message("user", "p")]
    node_chat(state, node, ctx)

    assert [m.role for m in state.conversation] == ["user", "assistant", "user"]
    assert state.conversation[1].content == "first draft"
    assert "improve your solution" in state.conversation[2].content
