"""Graph executor.

The pipeline is a directed graph walked as a state machine, not a DAG: the
verify -> judge -> correct -> verify feedback edge is a genuine cycle, and the
run ends on a guard rather than on running out of nodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Config, NodeConfig
from .nodes import NODE_TYPES, Context, NodeResult, apply_counters
from .state import RunState


@dataclass
class Outcome:
    status: str          # accept | fail | exhausted
    state: RunState
    solution: str | None = None

    @property
    def solved(self) -> bool:
        return self.status == "accept"


class GraphError(RuntimeError):
    pass


def route(node: NodeConfig, result: NodeResult) -> tuple[str | None, str | None, dict]:
    """-> (terminal, next_node, counter_updates)"""
    if result.terminal:
        return result.terminal, None, {}
    if result.next:
        return None, result.next, {}
    if result.label is not None and node.edges:
        edge = node.edges.get(result.label)
        if edge is None:
            raise GraphError(
                f"node '{node.name}' produced label '{result.label}' "
                f"but only defines edges {sorted(node.edges)}"
            )
        return edge.terminal, edge.next, edge.counters
    if node.next:
        return None, node.next, {}
    raise GraphError(f"node '{node.name}' has nowhere to go (no next, no matching edge, no terminal)")


def run(state: RunState, cfg: Config, ctx: Context, tracer=None, checkpoint=None) -> Outcome:
    pipe = cfg.pipeline
    if not state.current_node:
        state.current_node = pipe.start
        state.counters = dict(pipe.counters)

    while True:
        name = state.current_node
        node = pipe.nodes.get(name)
        if node is None:
            raise GraphError(f"undefined node '{name}'")

        started = time.time()
        if tracer:
            tracer.log(f">>>>>>> [{name}] {node.type}")
        result = NODE_TYPES[node.type](state, node, ctx)
        state = result.state
        elapsed = time.time() - started

        terminal, nxt, counters = route(node, result)
        apply_counters(state, counters)

        if tracer:
            tracer.record(
                {
                    "step": state.step,
                    "node": name,
                    "type": node.type,
                    "label": result.label,
                    "terminal": terminal,
                    "next": nxt,
                    "seconds": round(elapsed, 2),
                    "counters": dict(state.counters),
                    **result.usage,
                }
            )
            if node.into:
                tracer.log(f">>>>>>> [{name}] -> {node.into}")
                tracer.log(state.get(node.into, ""))
            if result.label is not None:
                tracer.log(f">>>>>>> [{name}] verdict: {result.label} {dict(state.counters)}")

        if terminal:
            if tracer:
                tracer.log(f">>>>>>> terminal: {terminal} after {state.step} steps {dict(state.counters)}")
            return Outcome(status=terminal, state=state, solution=state.get("solution") or None)

        state.current_node = nxt
        state.step += 1
        if checkpoint:
            checkpoint(state)

        if state.step > pipe.max_steps:
            if tracer:
                tracer.log(f">>>>>>> exhausted: exceeded max_steps={pipe.max_steps}")
            return Outcome(status="exhausted", state=state, solution=state.get("solution") or None)


def to_mermaid(cfg: Config) -> str:
    """Render the configured pipeline for `--graph`."""
    pipe = cfg.pipeline
    lines = ["flowchart TD"]
    shapes = {"chat": '["{n}<br/><i>chat</i>"]', "classify": '{{"{n}<br/><i>classify</i>"}}',
              "transform": '[/"{n}<br/><i>transform</i>"/]', "guard": '(["{n}<br/><i>guard</i>"])'}
    for name, node in pipe.nodes.items():
        lines.append(f"    {name}{shapes[node.type].format(n=name)}")
    terminals = set()

    def edge(src: str, dst: str | None, term: str | None, label: str = "") -> None:
        arrow = f" -->|{label}| " if label else " --> "
        if term:
            terminals.add(term)
            lines.append(f"    {src}{arrow}{term}")
        elif dst:
            lines.append(f"    {src}{arrow}{dst}")

    for name, node in pipe.nodes.items():
        if node.next:
            edge(name, node.next, None)
        for lbl, e in node.edges.items():
            edge(name, e.next, e.terminal, lbl)
        for rule in node.when:
            edge(name, rule.get("next"), None, rule["if"])
        if node.accept_if:
            edge(name, None, "accept", node.accept_if)
        if node.fail_if:
            edge(name, None, "fail", node.fail_if)
    for t in sorted(terminals):
        lines.append(f"    {t}(({t}))")
    lines.append(f"    start_(( )) --> {pipe.start}")
    return "\n".join(lines)
