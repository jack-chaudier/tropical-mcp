# Contributing

Thanks for contributing to `tropical-mcp`.

## Contribution boundary

This repository is source-available for evaluation, not an open-source fork-and-redistribute project.

- Pull requests and patch proposals to the canonical repository are welcome.
- Private local modifications are allowed for internal evaluation and for preparing upstream patches.
- Submitting a PR does not grant permission to redistribute modified copies outside the canonical repository.
- If you need broader derivative, redistribution, or production rights, contact `jackgaff@umich.edu`.

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
uv run --extra dev pytest
uv build
./scripts/validate_installed_wheel.sh
uv run tropical-mcp-full-validate
```

## MCP transport safety

Do not print to `stdout` from server/runtime code. MCP JSON-RPC owns stdout.
Use structured logging to `stderr` only.

## Change expectations

- Keep tool semantics backward compatible where possible.
- Add/adjust tests for any behavior change.
- Update README, docs, and CHANGELOG in the same PR when behavior changes.
- Use [`docs/RELEASE.md`](./docs/RELEASE.md) when a change affects tags, packaged artifacts, or the mirrored public evidence in `dreams`.
