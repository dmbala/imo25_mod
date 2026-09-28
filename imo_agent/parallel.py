"""Run N independent solver processes over one problem.

Success is read from the child's exit status (0 == accepted) and confirmed
against its JSONL trace, rather than grepping stdout for a magic sentence.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_child: subprocess.Popen | None = None
_handlers_installed = False

# Flags forwarded verbatim to each child solver.
PASSTHROUGH = ["model", "model_id", "base_url", "api_key_env", "effort", "max_tokens", "pipeline", "root"]


def _install_handlers() -> None:
    global _handlers_installed
    if _handlers_installed:
        return

    def forward(signum, _frame):
        try:
            if _child is not None and _child.poll() is None:
                try:
                    os.killpg(os.getpgid(_child.pid), signum)
                except Exception:
                    _child.terminate()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    _handlers_installed = True


def run_one(agent_id: int, problem: str, log_dir: str, timeout: int | None, other_prompts: list[str], overrides: dict):
    log_file = os.path.join(log_dir, f"agent_{agent_id:02d}.log")
    trace_file = os.path.join(log_dir, f"agent_{agent_id:02d}.jsonl")

    cmd = [sys.executable, "-m", "imo_agent.cli", problem,
           "--log", log_file, "--trace", trace_file, "--quiet"]
    # Only pass the flag when there is something to pass: an empty --other_prompts
    # used to reach the agent as a literal pair of quote characters.
    if other_prompts:
        cmd += ["--other_prompts", ",".join(other_prompts)]
    for key, value in overrides.items():
        if value is not None:
            cmd += [f"--{key.replace('_', '-')}", str(value)]

    try:
        _install_handlers()
        global _child
        _child = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
        )
        try:
            out, err = _child.communicate(timeout=timeout) if timeout else _child.communicate()
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(_child.pid), signal.SIGKILL)
            except Exception:
                _child.kill()
            return (agent_id, -1, "", f"timed out after {timeout}s", False)
        code = _child.returncode
        _child = None
        return (agent_id, code, out, err, code == 0 and _accepted(trace_file))
    except Exception as exc:
        return (agent_id, -1, "", f"failed: {exc}", False)


def _accepted(trace_file: str) -> bool:
    """Confirm the terminal record says accept; fall back to trusting exit 0."""
    path = Path(trace_file)
    if not path.exists():
        return True
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("terminal"):
            return rec["terminal"] == "accept"
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="imo-parallel", description="Run several solvers over one problem")
    p.add_argument("problem_file")
    p.add_argument("--num-agents", "-n", type=int, default=10)
    p.add_argument("--log-dir", "-d", default="logs")
    p.add_argument("--timeout", "-t", type=int, default=None)
    p.add_argument("--max-workers", "-w", type=int, default=None)
    p.add_argument("--other_prompts", "-o")
    p.add_argument("--exit-immediately", "-e", action="store_true")
    p.add_argument("--model")
    p.add_argument("--model-id")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--pipeline")
    p.add_argument("--root")
    args = p.parse_args(argv)

    os.makedirs(args.log_dir, exist_ok=True)
    problem = str(Path(args.problem_file).resolve())
    other_prompts = [s for s in (args.other_prompts or "").split(",") if s.strip()]
    overrides = {k: getattr(args, k) for k in PASSTHROUGH}

    print(f"Starting {args.num_agents} agents on {problem}")
    print(f"Model: {args.model or '(default from models.yaml)'}   Logs: {args.log_dir}")
    print("-" * 50)

    solved: list[int] = []
    failed: list[int] = []
    done = 0
    start = time.time()

    with ProcessPoolExecutor(max_workers=args.max_workers or args.num_agents) as pool:
        futures = {
            pool.submit(run_one, i, problem, args.log_dir, args.timeout, other_prompts, overrides): i
            for i in range(args.num_agents)
        }
        for future in as_completed(futures):
            agent_id, code, out, err, ok = future.result()
            done += 1
            if ok:
                solved.append(agent_id)
                print(f"\n*** Agent {agent_id:02d} found an accepted solution ***")
            elif code != 0:
                failed.append(agent_id)
                if err.strip():
                    print(f"[Agent {agent_id:02d}] {err.strip()[:400]}")
            print(f"Progress: {done}/{args.num_agents}")

            if ok and args.exit_immediately:
                print(f"Exiting early. Log: {os.path.join(args.log_dir, f'agent_{agent_id:02d}.log')}")
                for proc in list(getattr(pool, "_processes", {}).values()):
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                time.sleep(0.5)
                os._exit(0)

    print("\n" + "=" * 50)
    print(f"Elapsed: {time.time() - start:.1f}s   solved: {len(solved)}   errored: {len(failed)}")
    if solved:
        print(f"Accepted by agents: {sorted(solved)}")
        print(f"Logs in {os.path.abspath(args.log_dir)}")
    return 0 if solved else 1


if __name__ == "__main__":
    raise SystemExit(main())
