"""The functional units.

Every node is one function with the same contract:

    (RunState, NodeConfig, Context) -> NodeResult

which is what makes each one testable on its own against a stub client.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import expr, text as textlib
from .config import Config, MessageSpec, NodeConfig
from .state import Message, RunState
from .templates import render


@dataclass
class Context:
    cfg: Config
    clients: dict[str, object]
    default_model: str = ""

    def client(self, node: NodeConfig):
        name = node.model or self.default_model or self.cfg.default_model
        if name not in self.clients:
            raise KeyError(f"no client built for model '{name}'")
        return self.clients[name]


@dataclass
class NodeResult:
    state: RunState
    label: str | None = None
    terminal: str | None = None
    next: str | None = None
    usage: dict = field(default_factory=dict)


def _values(state: RunState, ctx: Context) -> dict[str, str]:
    """Placeholder namespace: prompt files first, state slots take precedence."""
    values: dict[str, str] = dict(ctx.cfg.prompts)
    values["problem"] = state.problem
    values.update(state.fields)
    return values


def _render(template: str, state: RunState, ctx: Context) -> str:
    return render(template, _values(state, ctx))


def _build(spec: MessageSpec, state: RunState, ctx: Context) -> list[Message]:
    if spec.each:
        items = getattr(state, spec.each, None)
        if items is None:
            items = state.get(spec.each, "")
            items = [items] if items else []
        return [Message(role=spec.role, content=_render(str(i), state, ctx)) for i in items if str(i).strip()]

    parts = []
    if spec.prompt:
        parts.append(ctx.cfg.prompts[spec.prompt])
    if spec.text:
        parts.append(spec.text)
    content = _render("\n\n".join(parts), state, ctx)
    return [Message(role=spec.role, content=content)]


def _messages(node: NodeConfig, state: RunState, ctx: Context) -> list[Message]:
    out: list[Message] = []
    for spec in node.messages:
        out.extend(_build(spec, state, ctx))
    return out


def _call(node: NodeConfig, state: RunState, ctx: Context, system: str, messages: list[Message]):
    client = ctx.client(node)
    result = client.complete(system, messages)
    body = result.text
    if getattr(client.cfg, "strip_preamble", False):
        body = textlib.strip_preamble(body)
    usage = {
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
    return body, usage


def node_chat(state: RunState, node: NodeConfig, ctx: Context) -> NodeResult:
    if node.reset:
        state.conversation = []
        state.system = _render(ctx.cfg.prompts[node.system], state, ctx) if node.system else ""
    state.conversation.extend(_messages(node, state, ctx))

    body, usage = _call(node, state, ctx, state.system, state.conversation)
    if node.into:
        state.set(node.into, body)
    return NodeResult(state=state, usage=usage)


def node_classify(state: RunState, node: NodeConfig, ctx: Context) -> NodeResult:
    # Scratch conversation: the classifier must not disturb the solver's thread.
    system = _render(ctx.cfg.prompts[node.system], state, ctx) if node.system else ""
    body, usage = _call(node, state, ctx, system, _messages(node, state, ctx))

    slot = node.into or node.name
    state.set(slot, body)
    label = _match(body, node.match)
    state.set(f"{slot}_label", label)
    return NodeResult(state=state, label=label, usage=usage)


def node_transform(state: RunState, node: NodeConfig, ctx: Context) -> NodeResult:
    source = state.get(node.source, "") if node.source else ""
    if node.fn == "split":
        value = textlib.split_marker(source, node.marker or "", node.keep)
    elif node.fn == "strip_preamble":
        value = textlib.strip_preamble(source)
    elif node.fn == "copy":
        value = source
    else:
        raise ValueError(f"node '{node.name}': unknown transform fn {node.fn!r}")
    state.set(node.into, value)
    return NodeResult(state=state)


def node_guard(state: RunState, node: NodeConfig, ctx: Context) -> NodeResult:
    apply_counters(state, node.counters)
    names = state.namespace()

    if node.accept_if and expr.evaluate(node.accept_if, names):
        return NodeResult(state=state, terminal="accept")
    if node.fail_if and expr.evaluate(node.fail_if, names):
        return NodeResult(state=state, terminal="fail")
    for rule in node.when:
        if expr.evaluate(rule["if"], names):
            return NodeResult(state=state, next=rule["next"])
    return NodeResult(state=state)


def apply_counters(state: RunState, updates: dict[str, str]) -> None:
    for name, raw in (updates or {}).items():
        token = str(raw).strip()
        if token.startswith(("+", "-")):
            state.counters[name] = state.counters.get(name, 0) + int(token)
        else:
            state.counters[name] = int(token)


def _match(body: str, table: dict[str, str]) -> str:
    haystack = (body or "").lower()
    for label, needle in table.items():
        if label == "default":
            continue
        if str(needle).lower() in haystack:
            return label
    return str(table.get("default", "no"))


NODE_TYPES = {
    "chat": node_chat,
    "classify": node_classify,
    "transform": node_transform,
    "guard": node_guard,
}
