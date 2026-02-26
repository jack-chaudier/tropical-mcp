# Contributing

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Quality gates

```bash
uv run --extra dev pytest -q
uv build
```

## MCP transport safety

Do not print to `stdout` from server/runtime code. MCP JSON-RPC owns stdout. Log only to `stderr`.
