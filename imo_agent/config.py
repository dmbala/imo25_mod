"""Load and validate configs/models.yaml + configs/pipeline.yaml."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import expr
from .llm.base import ModelConfig

NODE_TYPES = {"chat", "classify", "transform", "guard"}
APIS = {"anthropic", "openai_compat", "claude_cli"}
_COUNTER_UPDATE = re.compile(r"^[+-]?\d+$")


class _PipelineLoader(yaml.SafeLoader):
    """Keeps an explicit leading + on integers.

    Plain YAML reads `streak: +1` as the integer 1, which would silently turn a
    counter bump into an assignment and leave the graph looping until max_steps.
    """


def _construct_signed_int(loader, node):
    raw = loader.construct_scalar(node)
    if isinstance(raw, str) and raw.startswith("+"):
        return raw
    return yaml.SafeLoader.construct_yaml_int(loader, node)


_PipelineLoader.add_constructor("tag:yaml.org,2002:int", _construct_signed_int)
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    pass


@dataclass
class MessageSpec:
    role: str = "user"
    prompt: str | None = None       # name of a file in prompts/
    text: str | None = None         # inline template
    each: str | None = None         # expand one message per item of a state list


@dataclass
class EdgeConfig:
    label: str = ""
    next: str | None = None
    terminal: str | None = None
    counters: dict[str, str] = field(default_factory=dict)


@dataclass
class NodeConfig:
    name: str
    type: str
    model: str | None = None
    # chat / classify
    system: str | None = None
    messages: list[MessageSpec] = field(default_factory=list)
    reset: bool = False
    into: str | None = None
    match: dict[str, str] = field(default_factory=dict)
    # transform
    fn: str | None = None
    source: str | None = None
    marker: str | None = None
    keep: str = "after"
    # guard
    counters: dict[str, str] = field(default_factory=dict)
    accept_if: str | None = None
    fail_if: str | None = None
    when: list[dict] = field(default_factory=list)
    # routing
    edges: dict[str, EdgeConfig] = field(default_factory=dict)
    next: str | None = None


@dataclass
class Pipeline:
    start: str
    nodes: dict[str, NodeConfig]
    max_steps: int = 200
    counters: dict[str, int] = field(default_factory=dict)


@dataclass
class Config:
    models: dict[str, ModelConfig]
    default_model: str
    pipeline: Pipeline
    prompts: dict[str, str]
    root: Path


def expand_env(value):
    """Resolve ${VAR} and ${VAR:-default} inside config strings."""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            val = os.environ.get(m.group(1))
            if val:
                return val
            if m.group(2) is not None:
                return m.group(2)
            return ""
        return _ENV.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_prompts(prompts_dir: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for path in sorted(prompts_dir.glob("*.md")):
        prompts[path.stem] = path.read_text(encoding="utf-8")
    return prompts


def load_models(path: Path) -> tuple[dict[str, ModelConfig], str]:
    raw = expand_env(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    entries = raw.get("models") or {}
    if not entries:
        raise ConfigError(f"{path} defines no models")

    models: dict[str, ModelConfig] = {}
    for name, spec in entries.items():
        spec = dict(spec or {})
        api = spec.pop("api", None)
        if api not in APIS:
            raise ConfigError(f"model '{name}': api must be one of {sorted(APIS)}, got {api!r}")
        known = {f for f in ModelConfig.__dataclass_fields__ if f not in {"name", "api", "extra"}}
        extra = spec.pop("extra", None) or {}
        unknown = set(spec) - known
        if unknown:
            raise ConfigError(f"model '{name}': unknown keys {sorted(unknown)}")
        models[name] = ModelConfig(name=name, api=api, extra=extra, **spec)

    default = raw.get("default")
    if default and default not in models:
        raise ConfigError(f"default model '{default}' is not defined")
    return models, default or next(iter(models))


def _norm_label(value) -> str:
    """YAML 1.1 turns bare yes/no into booleans; map them back to labels."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)


def _message_specs(raw, node_name: str) -> list[MessageSpec]:
    specs = []
    for item in raw or []:
        if not isinstance(item, dict):
            raise ConfigError(f"node '{node_name}': each message must be a mapping, got {item!r}")
        unknown = set(item) - {"role", "prompt", "text", "each"}
        if unknown:
            raise ConfigError(f"node '{node_name}': unknown message keys {sorted(unknown)}")
        specs.append(MessageSpec(**item))
    return specs


def _edges(raw, node_name: str) -> dict[str, EdgeConfig]:
    edges = {}
    for label, spec in (raw or {}).items():
        spec = dict(spec or {})
        unknown = set(spec) - {"next", "terminal", "counters"}
        if unknown:
            raise ConfigError(f"node '{node_name}' edge '{label}': unknown keys {sorted(unknown)}")
        key = _norm_label(label)
        edges[key] = EdgeConfig(label=key, **spec)
    return edges


def load_pipeline(path: Path) -> Pipeline:
    raw = expand_env(yaml.load(path.read_text(encoding="utf-8"), Loader=_PipelineLoader) or {})
    nodes: dict[str, NodeConfig] = {}
    for name, spec in (raw.get("nodes") or {}).items():
        spec = dict(spec or {})
        ntype = spec.pop("type", None)
        if ntype not in NODE_TYPES:
            raise ConfigError(f"node '{name}': type must be one of {sorted(NODE_TYPES)}, got {ntype!r}")
        spec["messages"] = _message_specs(spec.get("messages"), name)
        spec["edges"] = _edges(spec.get("edges"), name)
        spec["match"] = {_norm_label(k): _norm_label(v) for k, v in (spec.get("match") or {}).items()}
        known = {f for f in NodeConfig.__dataclass_fields__ if f not in {"name", "type"}}
        unknown = set(spec) - known
        if unknown:
            raise ConfigError(f"node '{name}': unknown keys {sorted(unknown)}")
        nodes[name] = NodeConfig(name=name, type=ntype, **spec)

    if not nodes:
        raise ConfigError(f"{path} defines no nodes")
    start = raw.get("start") or next(iter(nodes))
    return Pipeline(
        start=start,
        nodes=nodes,
        max_steps=int(raw.get("max_steps", 200)),
        counters={k: int(v) for k, v in (raw.get("counters") or {}).items()},
    )


def load(root: Path, models_file: str = "models.yaml", pipeline_file: str = "pipeline.yaml") -> Config:
    root = Path(root)
    models, default = load_models(root / "configs" / models_file)
    pipeline = load_pipeline(root / "configs" / pipeline_file)
    prompts = load_prompts(root / "prompts")
    cfg = Config(models=models, default_model=default, pipeline=pipeline, prompts=prompts, root=root)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> list[str]:
    """Raise ConfigError on anything that would fail mid-run; return notes."""
    notes: list[str] = []
    pipe = cfg.pipeline
    names = set(pipe.nodes)

    if pipe.start not in names:
        raise ConfigError(f"start node '{pipe.start}' is not defined")

    # Names a guard may legally reference: declared counters plus anything a node writes.
    written = set(pipe.counters) | {"problem"}
    for node in pipe.nodes.values():
        if node.into:
            written.add(node.into)
        if node.type == "classify":
            written.add(f"{node.into or node.name}_label")
        for edge in node.edges.values():
            written |= set(edge.counters)
        written |= set(node.counters)

    def check_target(target: str | None, where: str) -> None:
        if target and target not in names:
            raise ConfigError(f"{where} points at undefined node '{target}'")

    for node in pipe.nodes.values():
        where = f"node '{node.name}'"
        if node.model and node.model not in cfg.models:
            raise ConfigError(f"{where}: model '{node.model}' is not defined in models.yaml")
        if node.system and node.system not in cfg.prompts:
            raise ConfigError(f"{where}: system prompt '{node.system}' not found in prompts/")
        for spec in node.messages:
            if spec.prompt and spec.prompt not in cfg.prompts:
                raise ConfigError(f"{where}: prompt '{spec.prompt}' not found in prompts/")
            if not (spec.prompt or spec.text or spec.each):
                raise ConfigError(f"{where}: a message needs one of prompt/text/each")

        check_target(node.next, where)
        for edge in node.edges.values():
            if not edge.next and not edge.terminal:
                raise ConfigError(f"{where} edge '{edge.label}': needs next or terminal")
            check_target(edge.next, f"{where} edge '{edge.label}'")
        for rule in node.when:
            check_target(rule.get("next"), f"{where} when-rule")
            if "if" not in rule:
                raise ConfigError(f"{where}: each when-rule needs an 'if'")

        for source, updates in [("node", node.counters)] + [
            (f"edge '{e.label}'", e.counters) for e in node.edges.values()
        ]:
            for counter, value in (updates or {}).items():
                if not _COUNTER_UPDATE.match(str(value)):
                    raise ConfigError(
                        f"{where} {source}: counter '{counter}' update {value!r} is not an integer "
                        f"or a signed change like \"+1\""
                    )

        if node.type == "transform":
            if not node.fn:
                raise ConfigError(f"{where}: transform needs fn")
            if not node.into:
                raise ConfigError(f"{where}: transform needs into")
        if node.type == "classify" and not node.match:
            raise ConfigError(f"{where}: classify needs a match table")
        if node.type in {"chat", "classify"} and not node.messages:
            raise ConfigError(f"{where}: {node.type} needs at least one message")

        for expression, label in (
            [(node.accept_if, "accept_if"), (node.fail_if, "fail_if")]
            + [(r.get("if"), "when") for r in node.when]
        ):
            if not expression:
                continue
            unknown = expr.identifiers(expression) - written
            if unknown:
                raise ConfigError(
                    f"{where} {label}: expression references {sorted(unknown)}, "
                    f"which no node writes and configs/pipeline.yaml does not declare under 'counters:'"
                )

    reachable = _reachable(pipe)
    orphans = sorted(names - reachable)
    if orphans:
        notes.append(f"unreachable nodes (never entered from '{pipe.start}'): {', '.join(orphans)}")
    return notes


def _reachable(pipe: Pipeline) -> set[str]:
    seen: set[str] = set()
    stack = [pipe.start]
    while stack:
        name = stack.pop()
        if name in seen or name not in pipe.nodes:
            continue
        seen.add(name)
        node = pipe.nodes[name]
        targets = [node.next] + [e.next for e in node.edges.values()]
        targets += [r.get("next") for r in node.when]
        stack += [t for t in targets if t]
    return seen
