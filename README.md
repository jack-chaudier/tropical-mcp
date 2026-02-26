# tropical-mcp

Production MCP server for **L2 tropical-algebra context compaction**.

This project addresses a long-context reliability failure mode: agents can remain "answer-valid" while silently switching the governing task intent under naive memory compression (the validity mirage).
`tropical-mcp` enforces a structural guard so load-bearing chunks (pivot + `k` predecessors) survive eviction when feasible, and emits auditable diagnostics when they cannot.
It is designed as a drop-in stdio MCP server for Claude Code, Codex, and similar tool-calling agents.

## Features

- `compact(messages, token_budget, policy, k)`
  - `l2_guarded` (default contract-guarded policy)
  - `l2_iterative_guarded` (iterative safe-removal variant)
  - `recency` (baseline)
- `inspect(messages, k)` for frontier + witness inspection
- `inspect_horizon(messages, k_max)` for feasible `k` range diagnostics
- `compact_auto(...)` for adaptive `k` selection
- `retention_floor(...)` for operational risk estimates
- `tag(messages)` for role inference diagnostics

## Docs

- Client config examples: [`docs/configuration.md`](./docs/configuration.md)
- Contribution guide: [`CONTRIBUTING.md`](./CONTRIBUTING.md)

## Project Layout

```text
tropical-mcp/
├── src/tropical_mcp/
│   ├── server.py
│   ├── algebra.py
│   ├── tagger.py
│   ├── compactor.py
│   └── benchmark_harness.py
├── tests/
├── scripts/
├── .github/workflows/ci.yml
├── pyproject.toml
└── README.md
```

## Install

```bash
cd /absolute/path/to/tropical-mcp
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Test + Validate

```bash
uv run --extra dev pytest -q
uv build
uv run tropical-mcp-full-validate
```

Or run the full script:

```bash
./scripts/full_validation.sh
```

## Run Server (stdio)

```bash
uv run tropical-mcp
```

or

```bash
python -m tropical_mcp
```

## Claude Code Registration

```bash
claude mcp add tropical-mcp --scope user -- \
  uv --directory /absolute/path/to/tropical-mcp run tropical-mcp
```

## Codex Registration

`~/.codex/config.toml`

```toml
[[mcp_servers]]
name = "tropical-mcp"
command = "uv"
args = ["--directory", "/absolute/path/to/tropical-mcp", "run", "tropical-mcp"]
```

## Replay Benchmark

```bash
uv run tropical-mcp-replay \
  --fractions 1.0,0.8,0.65,0.5,0.4 \
  --policies recency,l2_guarded \
  --k 3 \
  --line-count 200 \
  --output-dir artifacts/cyberops_mcp_replay
```

## Transport Safety

`stdout` is reserved for MCP JSON-RPC transport. Logging is sent only to `stderr`.
