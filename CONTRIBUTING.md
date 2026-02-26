# Contributing

Thanks for contributing to `tropical-mcp`.

## Development setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Quality gates (required before PR)

```bash
uv run --extra dev ruff check .
uv run --extra dev mypy src/tropical_mcp
uv run --extra dev pytest -q
uv build
uv run tropical-mcp-full-validate
```

## MCP transport safety

Do not print to `stdout` from server/runtime code. MCP JSON-RPC owns stdout.
Use structured logging to `stderr` only.

## Change expectations

- Keep tool semantics backward compatible where possible.
- Add/adjust tests for any behavior change.
- Update README, docs, and CHANGELOG in the same PR when behavior changes.
