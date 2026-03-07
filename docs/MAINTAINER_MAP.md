# Maintainer Map

Use this file as the shortest path to the place that should change when the public surface expands.

## Core implementation

- `src/tropical_mcp/` is the source of truth for tool behavior, runtime metadata, and validation helpers.
- `src/tropical_mcp/server.py` owns the MCP tool surface.
- `src/tropical_mcp/compactor.py`, `tagger.py`, and `algebra.py` hold the compaction logic and frontier math.

## Public fixtures and compatibility

- `src/tropical_mcp/fixtures/` holds the packaged golden fixtures that must survive wheel installs.
- `src/tropical_mcp/resources.py` is the helper for loading packaged fixtures without repo-local paths.
- If a fixture changes, update the corresponding tests in `tests/test_validation.py`, `tests/test_certificate.py`, and `tests/test_policy_invariance.py`.

## Validation and release checks

- `src/tropical_mcp/validation.py` defines the canonical functional validation report.
- `scripts/full_validation.sh` is the maintainer-facing quality gate.
- `scripts/validate_installed_wheel.sh` is the installed-artifact check that guards against wheel/repo drift.
- `.github/workflows/ci.yml` should mirror those checks closely.

## Docs and client examples

- `README.md` is the evaluator-facing landing page and should stay short.
- `docs/GUIDE.md` holds the deeper workflow explanation.
- `docs/configuration.md` holds client-registration details.
- `examples/codex/` contains the durable-memory and prompt bundle that should stay in sync with the docs.

## Public showcase sync

- `dreams/results/` is the public evidence bundle that mirrors the implementation validation outputs.
- When the public witness or validation shape changes, refresh the artifacts in `dreams/results/` and the explanatory copy in `dreams/site/`.
