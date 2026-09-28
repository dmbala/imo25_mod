# imo25_mod

A config-driven agent for olympiad problems. One solver runs against any model,
and the pipeline lives in YAML rather than in control flow.

Derived from [lyang36/IMO25](https://github.com/lyang36/IMO25) (Lin Yang, Yichen
Huang), which showed Gemini 2.5 Pro reaching gold-medal performance on IMO 2025
([arXiv:2507.15855](https://arxiv.org/abs/2507.15855)). The prompts here are
byte-identical to that project's; what changed is the plumbing around them.

## What is different

The original ships three near-identical agents — `agent.py`, `agent_oai.py`,
`agent_xai.py` — that differ only in how they build a request and read a
response. Adding a model means copying ~600 lines; changing a prompt means
editing three files; changing the loop means editing counters buried in an
`if` chain.

Here:

- **Any model, one interface.** Claude via the `anthropic` SDK; everything
  OpenAI-compatible — local vLLM/Ollama, Gemini-compat, OpenAI, xAI — via the
  `openai` SDK; and the `claude` CLI, which needs no API credential at all.
  A new hosted provider is an entry in `configs/models.yaml`.
- **The pipeline is data.** Nodes, edges, counters and stopping rules live in
  `configs/pipeline.yaml`.
- **Prompts are files.** `prompts/*.md`, edited without touching Python.
- **The agent is functional units.** Every node is one function with the
  signature `(RunState, NodeConfig, Context) -> NodeResult`, so each is testable
  on its own against a stub model.

## Install

```bash
cd imo25_mod
./setup.sh              # creates .venv and installs with uv
```

### Credentials for Claude

Being logged into **Claude Code is not enough**. Claude Code keeps its own OAuth
token in `~/.claude/.credentials.json`; the `anthropic` SDK never reads that
file. It resolves, first match wins: `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, then an OAuth profile under `~/.config/anthropic/`.

Pick one — or skip credentials entirely with `--model claude-code`, which
runs through the `claude` CLI and therefore through the login you already have
(see *Using your Claude Code login* below).

```bash
# 1. OAuth profile -- no static key to manage. Needs the `ant` CLI.
ant auth login                  # --no-browser on a headless login node
# the SDK then picks the profile up with no env var set

# 2. Hand this shell the active profile's short-lived token
set -a; eval "$(ant auth print-credentials --env)"; set +a

# 3. A plain API key
export ANTHROPIC_API_KEY=...
```

Two traps worth knowing. A profile is only consulted when **no** key is set, and
an empty `ANTHROPIC_API_KEY=""` still wins its slot and authenticates as an
empty key — truly `unset` it. And after `ant auth login`, Claude Code may warn
about a conflict between the profile and its own `/login`; keep one or the other.

Either way this is API usage, billed separately from a Claude Code
subscription. Run without credentials and the agent prints these options and
exits 2 rather than retrying.

`setup.sh` runs `uv sync` when that works and falls back to a local staging venv
when it does not — see *Cluster note* below. On an ordinary filesystem plain
`uv sync` is equivalent.

### Cluster note: installs on /n/netscratch can stall

`uv sync` works here, but on the netscratch (VAST) mount it is slow — ~80s with
a fully warm cache — and during this project's setup it stalled completely for
several minutes at a time, with no output and no CPU use, which looks exactly
like a hang. A `rm -rf .venv` also failed once with "Directory not empty" before
succeeding on a retry. Both symptoms point at the shared mount rather than at
uv, and neither reproduced reliably afterwards, so treat this as a caution and
not a diagnosis.

`setup.sh` therefore caps `uv sync` at 90 seconds and, if it does not finish,
resolves and installs into a staging venv on local disk before copying
`site-packages` across and rewriting the console-script shebangs. Pointing the
cache at local disk helps in any case:

```bash
export UV_CACHE_DIR=/tmp/uv-cache-$USER
```

## Use

```bash
uv run imo-solve problems/imo01.txt --model claude-opus-5 \
    --log run.log --trace run.jsonl

uv run imo-solve --list-models      # what is configured
uv run imo-solve --validate         # check the config, spend nothing
uv run imo-solve --graph            # print the pipeline as Mermaid
```

Against a model you serve yourself — no API key needed:

```bash
export IMO_BASE_URL=http://holygpu0000:8000/v1 IMO_MODEL=Qwen/Qwen3-32B
uv run imo-solve problems/imo01.txt --model local
```

Several attempts at once:

```bash
uv run imo-parallel problems/imo01.txt -n 10 --model claude-opus-5 -d logs/p1
```

Useful flags: `--model-id`, `--base-url`, `--api-key-env`, `--effort`,
`--max-tokens` override a registry entry ad hoc; `--pipeline other.yaml` swaps
the graph; `--checkpoint state.json --resume` survives a Slurm time limit by
restarting at the node the run died on.

### Using your Claude Code login

`--model claude-code` shells out to `claude --print` instead of calling an API,
so the work goes through your existing Claude Code login. No key, no `ant`, no
billing setup:

```bash
uv run imo-solve problems/imo01.txt --model claude-code
```

Set `model:` in the registry entry to an alias (`opus`, `sonnet`, `fable`) or a
full model name. The adapter runs with `--restricted` (no Bash/REPL tools),
`--permission-prompts none` (anything that would prompt is denied) and
`--strict-mcp-config` (your MCP servers are ignored), in a scratch working
directory so no `CLAUDE.md` leaks into the prompt.

**What it costs you.** Claude Code is an agent harness, not a completion
endpoint, so its own context rides along on every call: measured here, one
request returning three tokens still billed ~14,900 input tokens. A full solve
makes dozens of calls, so this route is substantially dearer and slower than the
same model over the API. Print mode is also single-turn, so multi-turn nodes
(`improve`, `correct`) are flattened into a `## User` / `## Assistant`
transcript rather than sent as real turns.

Use it to try the pipeline without setting up billing; use `--model
claude-opus-5` for real runs.

## The pipeline

For how the code is structured and why — the node contract, the executor, the
provider layer, the config traps and the trade-offs — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```
solve ──▶ improve ──▶ extract ──▶ verify ──▶ judge ──▶ check ──▶ decide
                         ▲                                         │
                         └── correct ◀── bugreport ◀── penalize ◀──┘
```

This is **not a DAG**: `judge → … → verify` is a real cycle, and a run ends when
a guard fires (`streak >= 5` accepts, `errors >= 10 or iterations >= 30` fails),
not when the nodes run out. The executor is correspondingly small — walk
`current → next` until a terminal.

Four node types:

| Type | Does |
|---|---|
| `chat` | Build or extend a conversation and call the model. `reset: true` starts fresh; `reset: false` continues the existing thread. |
| `classify` | Short call whose reply is matched to a label that picks the outgoing edge. |
| `transform` | Pure text surgery — no model call. |
| `guard` | Update counters, evaluate `accept_if` / `fail_if` / `when`. No model call. |

Counters are updated on edges and guards, so the thresholds that were magic
numbers upstream are three lines of config:

```yaml
check:
  type: guard
  counters: {iterations: +1}
  accept_if: "streak >= 5"
  fail_if: "errors >= 10 or iterations >= 30"
  next: decide
```

Guard expressions are evaluated through an AST whitelist — comparisons, boolean
operators, names, literals — not `eval`.

### Adding a model

```yaml
  deepseek-reasoner:
    api: openai_compat
    model: deepseek-reasoner
    base_url: https://api.deepseek.com/v1
    key_env: DEEPSEEK_API_KEY
    max_tokens: 32000
    max_tokens_param: max_tokens
```

`${VAR}` and `${VAR:-default}` are resolved from the environment at load time,
which is how the `local` entry follows your vLLM server around.

Per-model knobs worth knowing: `effort` maps to Anthropic's `output_config.effort`
and to `reasoning_effort` on OpenAI-compatible servers; `temperature` is only
sent when set, because current Claude models reject it; `max_tokens_param`
exists because compat servers disagree on `max_tokens` vs `max_completion_tokens`;
`strip_preamble` drops chatter before `**Summary**` for models that produce it.

## Output

`--log` writes the human-readable transcript. `--trace` writes one JSON record
per node execution:

```bash
jq -r '[.step, .node, .label // "-", .output_tokens, .seconds] | @tsv' run.jsonl
```

## Tests

```bash
uv run pytest
```

No network and no API key: nodes run against a stub model. The suite covers the
text transforms, guard expressions, config validation, both terminal paths
through the real pipeline, and a check that every prompt is still byte-identical
to the upstream original.

## Cost

A full run is dozens of long reasoning calls — solve, improve, then verify and
correct up to thirty times. `--validate`, `--graph` and `pytest` are free;
everything else is not. Start with one run before fanning out.

## Licence

MIT, carried from the upstream project. See `LICENSE`.
