# Host-agent inference

`--agent` sends **inference** to a local CLI harness (for example Claude
Code) while OCR keeps the rest of the review pipeline: file selection,
grouping, the multi-round tool loop, the reflection filter, comment
re-location, sessions, coverage, resume, SARIF, and the viewer.

It is not `ocr delegate`. That command prints a review spec and stops.
The host agent then runs the whole review itself. OCR does not call an
LLM, does not loop on tools, and does not write a session. Use
`ocr delegate` when you want the harness to own the workflow. Use
`--agent` when you want OCR to own the workflow and only borrow the
harness as a model.

Both are legitimate. They solve different problems.

## Setup

The harness command is read only from `~/.opencodereview/config.json`,
never from a repository. OCR reviews untrusted trees; a repo-local
command would be code execution on the reviewer's machine.

A model is required. `--agent` takes it from `--model`, or from the
top-level `model` field in that same config file. `command` alone is
not enough.

```bash
ocr config set host_agents.claude-code.command claude
ocr config set model claude-opus-4-6
ocr llm test --agent claude-code
ocr review --agent claude-code --from main --to HEAD
ocr scan --agent claude-code
```

`--agent` and `--provider` are mutually exclusive. OCR rejects the
combination rather than picking one.

### Fields

| Key | Meaning |
| --- | --- |
| `host_agents.<name>.command` | Executable to spawn. Required. |
| `host_agents.<name>.args` | Extra argv, JSON array. Optional. |
| `host_agents.<name>.env` | Extra environment, JSON array of `KEY=VALUE`. Optional. |

`args` is prepended to the argv OCR already builds (`--bare`, `-p`,
`--output-format json`, `--json-schema <tempfile>`, `--tools ""`, plus
`--model` and `--system-prompt` when set, and `--session-id` or
`--resume` once a conversation id exists). Do not repeat those flags.
Values that start with `-` need a `--` separator so `ocr config set`
does not treat them as its own flags. `--foo` below is a placeholder
for extra harness flags you actually need:

```bash
ocr config set host_agents.claude-code.args -- '["--foo"]'
ocr config set host_agents.claude-code.env -- '["FOO=bar"]'
```

Remove an entry with `ocr config unset host_agents.claude-code`.

Resulting `~/.opencodereview/config.json` shape:

```json
{
    "model": "claude-opus-4-6",
    "host_agents": {
        "claude-code": {
            "command": "claude",
            "args": ["--foo"],
            "env": ["FOO=bar"]
        }
    }
}
```

Omit `args` and `env` when empty.

## Limitations

This path does not provide several things the managed HTTP providers
do.

**Tool calls are synthesized, not native.** The harness is invoked with
`--tools ""` and a JSON Schema. It cannot emit OCR tool-call JSON. OCR
asks for schema-validated structured output and builds tool calls from
that. Review quality under this path is not yet measured against the
managed path.

**`max_tokens` is advisory.** The Claude Code CLI exposes no
`--max-tokens` flag (verified against `claude --help` on v2.1.274), so
the per-request cap is not enforced on this transport. The aggregate
`--max-tokens-budget` still applies.

**Token usage may be estimated.** When the harness reports a `usage`
object, OCR passes those counts through. When it reports none, OCR
estimates prompt and completion tokens with tiktoken so the budget
still functions. Estimated numbers are not billing-accurate. Cache
counts stay 0 in the estimate.

**No provider replay state.** Thinking blocks and provider-signed
reasoning cannot survive the round trip, so they are not reused across
turns.

**The retry report is empty.** It describes HTTP attempts. This
transport makes none. `ocr review` still runs; the report has nothing
to list.

**Prompt caching depends on session continuity.** For each main-task
loop OCR generates a conversation UUID. The first CLI call for that id
passes `--session-id` so the harness creates a session under OCR's id;
later rounds pass `--resume` with the same id. A harness that does not
honor those flags will re-read context each round.

**Check your harness's terms.** Driving a subscription-authenticated
CLI as an inference backend for a third-party tool may not be
permitted. That is the user's call, and it can only be made if it is
stated.

## Gateway alternative

If the harness or an internal gateway exposes an OpenAI- or
Anthropic-compatible HTTP endpoint, point OCR at it directly instead
of using `--agent`:

```bash
export OCR_LLM_URL=https://gateway.example.com/v1
export OCR_LLM_TOKEN=...
export OCR_LLM_MODEL=...
ocr llm test
```

A configured provider (`ocr config set provider ...` plus
`providers.<name>.api_key` or the provider's env var) is the same
shape of path.

That path gets native tool calls, real provider-reported usage, replay
state, and the retry report — everything the list above gives up. It
needs a URL and a token, so it does not satisfy the "use my existing
subscription, no API key" goal, which is the whole trade.

This is existing behavior, not a new feature.
