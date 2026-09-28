"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config as config_mod
from . import graph as graph_mod
from .llm import build_client
from .llm.base import AuthError
from .nodes import Context
from .state import RunState
from .trace import Tracer, load_checkpoint, save_checkpoint

DEFAULT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imo-solve",
        description="Solve an olympiad problem with a config-driven agent graph.",
    )
    p.add_argument("problem_file", nargs="?", help="path to the problem statement")

    p.add_argument("--model", help="model name from configs/models.yaml")
    p.add_argument("--model-id", help="override the wire model string")
    p.add_argument("--base-url", help="override the endpoint (e.g. a vLLM server)")
    p.add_argument("--api-key-env", help="override which env var holds the key")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-tokens", type=int)

    p.add_argument("--root", default=str(DEFAULT_ROOT), help="dir holding configs/ and prompts/")
    p.add_argument("--pipeline", default="pipeline.yaml", help="pipeline file in configs/")
    p.add_argument("--models-file", default="models.yaml", help="model registry in configs/")

    p.add_argument("--log", "-l", help="write the human-readable log here")
    p.add_argument("--trace", help="write one JSON record per node here")
    p.add_argument("--quiet", "-q", action="store_true", help="do not echo the log to stdout")
    p.add_argument("--other_prompts", "-o", help="extra user prompts, comma separated")
    p.add_argument("--max-runs", "-m", type=int, default=1, help="retry the whole graph N times")
    p.add_argument("--checkpoint", help="save run state here after every step")
    p.add_argument("--resume", "-r", action="store_true", help="resume from --checkpoint")

    p.add_argument("--validate", action="store_true", help="check the config and exit")
    p.add_argument("--graph", action="store_true", help="print the pipeline as Mermaid and exit")
    p.add_argument("--list-models", action="store_true", help="list configured models and exit")
    return p


def apply_overrides(cfg: config_mod.Config, args) -> str:
    name = args.model or cfg.default_model
    if name not in cfg.models:
        raise SystemExit(f"model '{name}' is not in configs/{args.models_file}; known: {', '.join(sorted(cfg.models))}")
    model = cfg.models[name]
    if args.model_id:
        model.model = args.model_id
    if args.base_url:
        model.base_url = args.base_url
    if args.api_key_env:
        model.key_env = args.api_key_env
    if args.effort:
        model.effort = args.effort
    if args.max_tokens:
        model.max_tokens = args.max_tokens
    return name


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)

    try:
        cfg = config_mod.load(root, args.models_file, args.pipeline)
    except config_mod.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.list_models:
        for name, model in sorted(cfg.models.items()):
            mark = "*" if name == cfg.default_model else " "
            if model.api == "claude_cli":
                target = f"{model.extra.get('binary', 'claude')} CLI (Claude Code login)"
            else:
                target = model.base_url or "api.anthropic.com"
            print(f"{mark} {name:<18} {model.api:<14} {model.model:<24} {target}")
        return 0

    if args.graph:
        print(graph_mod.to_mermaid(cfg))
        return 0

    if args.validate:
        notes = config_mod.validate(cfg)
        print(f"ok: {len(cfg.pipeline.nodes)} nodes, {len(cfg.models)} models, {len(cfg.prompts)} prompts")
        print(f"    start={cfg.pipeline.start} max_steps={cfg.pipeline.max_steps}")
        for note in notes:
            print(f"    note: {note}")
        return 0

    if not args.problem_file:
        print("a problem file is required (or use --validate / --graph / --list-models)", file=sys.stderr)
        return 2

    model_name = apply_overrides(cfg, args)
    needed = {model_name} | {n.model for n in cfg.pipeline.nodes.values() if n.model}
    try:
        clients = {name: build_client(cfg.models[name]) for name in needed}
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ctx = Context(cfg=cfg, clients=clients, default_model=model_name)

    problem = Path(args.problem_file).read_text(encoding="utf-8")
    other_prompts = [s for s in (args.other_prompts or "").split(",") if s.strip()]

    tracer = Tracer(args.log, args.trace, echo=not args.quiet)
    checkpoint = (lambda st: save_checkpoint(args.checkpoint, st)) if args.checkpoint else None

    try:
        tracer.log(f">>>>>>> model: {model_name} ({cfg.models[model_name].model})")
        tracer.log(f">>>>>>> pipeline: {args.pipeline} start={cfg.pipeline.start}")
        if other_prompts:
            tracer.log(f">>>>>>> other prompts: {other_prompts}")

        for attempt in range(args.max_runs):
            tracer.log(f">>>>>>> run {attempt} of {args.max_runs}")
            if args.resume and args.checkpoint and Path(args.checkpoint).exists():
                state = load_checkpoint(args.checkpoint)
                tracer.log(f">>>>>>> resumed at node '{state.current_node}' step {state.step}")
            else:
                state = RunState(problem=problem, other_prompts=other_prompts)

            try:
                outcome = graph_mod.run(state, cfg, ctx, tracer=tracer, checkpoint=checkpoint)
            except AuthError as exc:   # retrying cannot fix credentials
                tracer.log(f">>>>>>> {exc}")
                return 2
            except Exception as exc:   # a run may die on a transient API failure
                tracer.log(f">>>>>>> run {attempt} failed: {type(exc).__name__}: {exc}")
                continue

            if outcome.solved:
                tracer.log(f">>>>>>> accepted solution on run {attempt}")
                tracer.log(outcome.solution or "")
                return 0
            tracer.log(f">>>>>>> run {attempt} ended: {outcome.status}")

        tracer.log(">>>>>>> no accepted solution")
        return 1
    finally:
        tracer.close()


if __name__ == "__main__":
    raise SystemExit(main())
