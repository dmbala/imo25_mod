# Architecture

How `imo25_mod` is put together, and why. For usage see [`../README.md`](../README.md).

## 1. What problem the structure solves

The upstream project ([lyang36/IMO25](https://github.com/lyang36/IMO25)) ships three
agents — `agent.py`, `agent_oai.py`, `agent_xai.py` — at 656, 568 and 642 lines.
They are ~85% identical. All three carry the same five prompt strings and the same
solve → verify → correct loop, and differ only in three functions
(`build_request_payload`, `send_api_request`, `extract_text_from_response`) plus how
each appends a conversation turn.

That shape has three costs:

| Want to… | Upstream cost |
|---|---|
| add a model | copy ~600 lines |
| change a prompt | edit three files, hope they stay in sync |
| change the control flow | hand-edit counter bookkeeping inside an `if` chain |
| test the loop | impossible without hitting a live API |

Everything below follows from wanting those four to be cheap. The whole package is
~1,550 lines.

## 2. Layers

```
                 cli.py / parallel.py        entry points
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
   config.py          graph.py           trace.py         load, execute, observe
   expr.py                │
       │                  │
       │              nodes.py                             the functional units
       │            ┌─────┴─────┐
       │      templates.py   text.py                       pure helpers
       │                  │
       └──────────────  llm/  ──────────────┐              provider adapters
                  base · anthropic_client
                  openai_compat · claude_cli
```

Dependencies point downward only. `llm/` knows nothing about graphs, `expr.py` and
`text.py` know nothing about anything, and `graph.py` never mentions a provider.

## 3. Core abstraction: the node contract

Every unit of work is one function with one signature:

```python
def node(state: RunState, cfg: NodeConfig, ctx: Context) -> NodeResult
```

That uniformity is what makes the executor trivial (§5) and the units testable
against a stub client with no network.

### RunState — everything a run knows

```python
@dataclass
class RunState:
    problem: str                     # the problem statement
    other_prompts: list[str]         # extra user turns from --other_prompts
    system: str                      # rendered system prompt of the live conversation
    conversation: list[Message]      # [{"role": "user"|"assistant", "content": str}]
    fields: dict[str, str]           # named slots: solution, detailed, verification, …
    counters: dict[str, int]         # streak, errors, iterations
    current_node: str
    step: int
```

`fields` is a dict rather than named attributes on purpose: a node declares
`into: <name>` in YAML and that slot springs into existence. Adding a node never
means editing the dataclass. The whole object is JSON round-trippable
(`to_json` / `from_json`), which is what makes checkpoint/resume a four-line feature.

### NodeResult — what a node reports back

```python
@dataclass
class NodeResult:
    state: RunState
    label: str | None = None      # classify: picks the outgoing edge
    terminal: str | None = None   # guard: ends the run ("accept" / "fail")
    next: str | None = None       # guard: an explicit jump
    usage: dict                   # tokens, model, for the trace
```

A node never decides its own successor beyond these hints; routing is resolved
centrally in `graph.route`.

## 4. The four node types

| Type | Model call | Purpose | Replaces upstream |
|---|---|---|---|
| `chat` | yes | build/extend a conversation, write the reply into a slot | `init_explorations`, the correction branch |
| `classify` | yes | short call whose reply is matched to a label | the `"yes" not in o.lower()` checks |
| `transform` | no | pure text surgery | `extract_detailed_solution`, `extract_solution` |
| `guard` | no | update counters, evaluate `accept_if`/`fail_if`/`when` | `correct_count >= 5`, `error_count >= 10` |

Two details carry real weight:

**`reset: true` vs `reset: false`.** This is the difference between starting a fresh
conversation (`verify`, with the grader system prompt) and continuing the existing
one (`improve`, which appends to the solver's thread). Upstream this distinction is
implicit in *which function you happen to be reading*; here it is one declared field.

**`classify` is scratch.** It builds a throwaway message list and never touches
`state.conversation`, so asking "is this solution correct?" cannot pollute the
solver's thread. `chat` with `reset: true` does replace the conversation — safe
because by then the solver's thread is no longer needed, exactly as upstream rebuilds
its payload from scratch in the correction branch.

### Messages are declared, not coded

One mechanism covers all four chat-shaped nodes:

```yaml
messages:
  - {role: user, text: "{problem}"}
  - {role: user, each: other_prompts}      # one message per list item
  - {role: assistant, text: "{solution}"}  # replay a previous output
  - {role: user, prompt: correction, text: "{bug_report}"}   # joined with \n\n
```

`prompt:` names a file in `prompts/`; `text:` is an inline template; giving both
joins them with a blank line, which reproduces how upstream concatenated the
correction prompt with the bug report.

## 5. The executor

`graph.run` is the entire control flow:

```python
while True:
    node   = pipe.nodes[state.current_node]
    result = NODE_TYPES[node.type](state, node, ctx)   # the functional unit
    terminal, nxt, counters = route(node, result)
    apply_counters(state, counters)
    if terminal:
        return Outcome(terminal, state)
    state.current_node = nxt
    state.step += 1
    if checkpoint: checkpoint(state)
    if state.step > pipe.max_steps:
        return Outcome("exhausted", state)
```

`route` resolves successors by precedence: an explicit `terminal`, then an explicit
`next`, then a label-selected edge, then the node's default `next`. Falling off the
end raises `GraphError` naming the node — a config mistake fails loudly rather than
looping.

### It is a state machine, not a DAG

```
solve ─▶ improve ─▶ extract ─▶ verify ─▶ judge ─▶ check ─▶ decide
                                 ▲                  │         │
                                 └── correct ◀── bugreport ◀── penalize
```

`judge → … → verify` is a genuine cycle, so the graph is not acyclic, and a run ends
when a **guard fires**, not when nodes run out. Three shapes were considered:

| Option | Verdict |
|---|---|
| unroll the loop into a true DAG | 30 unrolled stages; the plan gets large and the loop stops being visible |
| hide the loop inside one `refine` node | top level becomes acyclic but the interesting part becomes opaque |
| **cyclic graph + guards** | **chosen** — faithful to the algorithm, every step is a first-class node, the executor stays ~20 lines |

## 6. Faithfulness to the original

The pipeline reproduces upstream semantics exactly, including one subtlety worth
stating because it looks like an off-by-one:

> Upstream increments `error_count` at the **top** of the loop (acting on the
> *previous* verdict) and tests `error_count >= 10` at the **bottom** (after the
> *next* verify). So one full correct→verify→judge round always happens between the
> increment and the check.

The `penalize` node placement reproduces that: `errors` is bumped on the way *into* a
correction, and `check` tests it only after the following verify. Getting this wrong
would silently cost one correction attempt per run.

Similarly `counters.streak` starts at **1**, matching `correct_count = 1` before
upstream's loop begins.

| Upstream (`agent.py`) | Here |
|---|---|
| `step1_prompt` etc. (5 strings) | `prompts/*.md`, byte-identical |
| `init_explorations` | `solve` + `improve` |
| `verify_solution` | `extract` + `verify` + `judge` + `bugreport` |
| `correct_count >= 5` | `check.accept_if` |
| `error_count >= 10`, `range(30)` | `check.fail_if` |
| `for i in range(max_runs)` | `--max-runs` in `cli.main` |
| `save_memory` / `load_memory` | `--checkpoint` / `--resume` (whole state, not a subset) |

Prompt fidelity is enforced by a test that re-parses the upstream file with `ast` and
compares byte-for-byte, skipping when the checkout is absent.

## 7. Configuration

Three inputs, all data:

- **`configs/models.yaml`** — the model registry. A new hosted provider is an entry,
  not code.
- **`configs/pipeline.yaml`** — nodes, edges, counters, guards.
- **`prompts/*.md`** — prompt bodies, referenced by filename stem.

`${VAR}` and `${VAR:-default}` are expanded from the environment at load time, which
is how the `local` entry follows a vLLM server around without edits.

### Two YAML traps, handled in the loader

**Bare `yes:` is a boolean.** YAML 1.1 parses unquoted `yes`/`no` keys as `True`/`False`,
so edge and match labels would arrive as `"True"`. `_norm_label` maps them back, and
both `"yes"` and bare `yes` work.

**Bare `+1` is the integer 1.** This one is nastier: `streak: +1` would *assign* 1
instead of incrementing, the streak would never grow, and every run would silently
burn to `max_steps` with no error. A custom `_PipelineLoader` preserves an explicit
leading `+` as a string, so both `+1` and `"+1"` mean "increment". `validate` then
rejects any counter update that is not an integer or a signed change.

*(This was a real bug during development, found only because four graph tests
returned `exhausted`.)*

### Validation before spending money

`config.validate` is deliberately strict, because the alternative is discovering a
typo forty minutes and several dollars into a run. It rejects: unknown node types and
apis, `next`/`else` targets that do not exist, prompt references that do not resolve,
model references that are not registered, edges with neither `next` nor `terminal`,
malformed counter updates, and guard expressions naming something no node writes and
no `counters:` block declares. It reports unreachable nodes as a note. `--validate`
runs it and exits; `--graph` emits Mermaid. Both cost nothing.

### Guard expressions

Conditions like `errors >= 10 or iterations >= 30` are evaluated by `expr.py` through
an **AST whitelist** — comparisons, boolean operators, `not`, names and literals —
never `eval`. Pipeline configs get shared and hand-edited; a config file should not be
able to execute code. Unknown names raise `ExprError` listing what *is* available,
which turns a typo into a readable message instead of a silent `False`.

## 8. Provider layer

The canonical interface keeps the **system prompt separate from the message list**:

```python
class LLMClient(Protocol):
    cfg: ModelConfig
    def complete(self, system: str, messages: list[Message]) -> Completion: ...
```

That is not cosmetic. Both target APIs want it that way (Anthropic `system=`, Gemini
`systemInstruction`), and upstream's `agent_oai.py` flattened it into
`f"System: {…}\n\nUser: {…}"` — a lossy workaround this design avoids.

`ModelConfig` is one flat dataclass covering all three adapters; each reads the fields
it understands and ignores the rest.

| Adapter | Covers | Notable handling |
|---|---|---|
| `anthropic_client` | Claude | streaming + `get_final_message()`, adaptive thinking, `output_config.effort`, refusal detection, graceful `fallbacks` degradation |
| `openai_compat` | OpenAI, xAI, DeepSeek, OpenRouter, Gemini-compat, vLLM/Ollama | `max_tokens_param`, conditional `reasoning_effort`, `api_key="EMPTY"` for local servers |
| `claude_cli` | Claude Code login | subprocess `claude --print --output-format json`, single-turn flattening, tools stripped |

Quirks worth knowing, each encoded as a config field rather than a branch in shared code:

- **`temperature` is only sent when set.** Current Claude models reject it outright.
- **`max_tokens_param`.** OpenAI wants `max_completion_tokens`; vLLM, Ollama and
  Gemini-compat want `max_tokens`.
- **`reasoning_effort` is only sent when configured.** Non-reasoning models and many
  compat servers 400 on it.
- **Empty API keys are never passed.** An empty string still wins the SDK's precedence
  slot and authenticates as an empty key, shadowing a valid OAuth profile — so the
  Anthropic adapter passes a key only when there is a real one and otherwise lets the
  SDK resolve `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile.
- **Streaming is not optional on Claude.** `max_tokens` is 32000 and a hard proof
  generates for minutes; `create()` would risk the HTTP timeout.

`AuthError` is separate from `RuntimeError` so `cli.main` can treat it as fatal —
retrying a run cannot conjure credentials.

### The `claude_cli` trade-off

It needs no API credential, which is the entire point, but Claude Code is an agent
harness rather than a completion endpoint. Measured on the imo01 run: **~20k input
tokens per call of harness overhead**, near-constant whether the call emits 24,509
output tokens or 4. Print mode is also single-turn, so multi-turn nodes are flattened
into a `## User` / `## Assistant` transcript. Good for trying the pipeline without
billing; use the API adapter for real work.

## 9. Templating

`templates.render` substitutes `{name}` via regex, **not** `str.format`. Every prompt
here is full of TeX — `$x_{i}$`, `\frac{a}{b}` — which `str.format` would choke on or
silently eat. Unknown placeholders are left untouched for the same reason.

The namespace is prompt bodies first, then state slots (so state wins on collision),
which is how `verify_user.md` can interpolate `{verify_reminder}`, another prompt
file, alongside `{problem}` and `{detailed}`.

## 10. Observability

- **`--log`** — human transcript, `>>>>>`-prefixed lines timestamped, matching
  upstream's log style.
- **`--trace`** — one JSON record per node execution: step, node, type, label,
  terminal, seconds, counters, model, token counts. Runs become analyzable rather
  than greppable.
- **`--checkpoint`** — the whole `RunState` serialized after every step, so `--resume`
  restarts at the node that died. This matters under Slurm time limits, and it is a
  strict improvement on upstream's partial memory file.

`parallel.py` reads success from the child's **exit status**, confirmed against the
terminal record in its trace. Upstream grepped stdout for the literal string
`"Found a correct solution in run"` — reliable only until someone rewords a log line.

## 11. Testing

44 tests, no network, no API key. The design is what makes this possible: nodes take a
client through `Context`, so a stub answering by role substitutes for a model.

| Area | Approach |
|---|---|
| `text`, `expr`, `templates` | direct unit tests — pure functions |
| `config` | validation rejects each class of broken config |
| nodes | individual units against a stub client |
| graph | full walks asserting both terminals, counter arithmetic, and that a rejection routes through `bugreport` → `correct` |
| `claude_cli` | a fake `claude` executable recording argv and returning canned JSON |
| prompt fidelity | byte-comparison against the upstream `agent.py` |

The graph tests are the valuable ones: they pin the accept path (streak 1→5), the fail
path (the eleventh rejection, not the tenth — §6), and the streak reset.

## 12. Extending it

**Add a hosted model** — an entry in `models.yaml`. No code:

```yaml
  deepseek-reasoner:
    api: openai_compat
    model: deepseek-reasoner
    base_url: https://api.deepseek.com/v1
    key_env: DEEPSEEK_API_KEY
    max_tokens_param: max_tokens
```

**Change the pipeline** — edit `pipeline.yaml`, run `--validate` and `--graph`. The
commented-out `recheck` node is a worked example: upstream wrote a "re-review the bug
report" step and then commented it out (`agent.py:398-415`); here, enabling that
experiment is a config edit.

**Add a node type** — write `node_x(state, cfg, ctx) -> NodeResult`, register it in
`nodes.NODE_TYPES`, add the name to `config.NODE_TYPES` and any new fields to
`NodeConfig`.

**Add a provider** — implement `complete(system, messages) -> Completion`, register it
in `llm.base.build_client` and in `config.APIS`.

## 13. Known limitations

- **No per-node retry.** A transient API failure aborts the run; `--max-runs` restarts
  from the beginning and `--checkpoint`/`--resume` recovers, but a node-level retry
  with backoff would be better. The SDKs' own `max_retries=5` covers 429/5xx only.
- **`strip_preamble` uses `rfind("Summary")`**, inherited from `agent_xai.py`. It can
  over-trim a solution that says "Summary" late, which is why it is opt-in per model.
- **The verifier judges, not ground truth.** `accept` means five consecutive passes
  from a grader persona reading the solution cold. That is a proxy for correctness, not
  a proof of it.
- **`classify` matching is substring-based**, so a verdict prose-ing its way to "yes"
  inside a longer sentence counts as a pass. Upstream had the same property.
- **Parallel fan-out multiplies cost linearly** and there is no shared budget ceiling
  across agents.
