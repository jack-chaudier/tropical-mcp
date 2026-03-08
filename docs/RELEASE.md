# Release Checklist

Use this checklist before cutting a public `tropical-mcp` release.

## Versioning and metadata

1. Update `pyproject.toml` version.
2. Add release notes to `CHANGELOG.md`.
3. Refresh `CITATION.cff` if the release date or preferred citation context changed.
4. Confirm `README.md`, `docs/GUIDE.md`, `docs/configuration.md`, `docs/ARCHITECTURE.md`, and `docs/METHODOLOGY.md` still match the supported workflow and boundaries.

## Validation gate

Run the full maintainer gate from a clean checkout:

```bash
uv run --extra dev ruff check .
uv run --extra dev mypy src/tropical_mcp
uv run --extra dev pytest
uv build
./scripts/validate_installed_wheel.sh
uv run tropical-mcp-full-validate
./scripts/full_validation.sh
```

## Integrity and security sweep

1. Run `./scripts/update_release_checksums.sh`.
2. Confirm `examples/codex/SHA256SUMS.txt` and `src/tropical_mcp/fixtures/SHA256SUMS.txt` changed only when their underlying files changed.
3. Confirm examples, fixtures, docs, and logs contain no secrets or machine-local paths.
4. Confirm the README and configuration docs still distinguish the minimum smoke flow from the fuller research workflow.

## Public artifact sync

1. Regenerate the mirrored validation outputs in `dreams/results/`.
2. If fixture refs or report shape changed, update `dreams/scripts/validate_artifacts.py`.
3. If the replay witness or recommended workflow changed, refresh `dreams/site/data_miragekit.json`, `dreams/site/index.html`, `dreams/site/evidence.html`, and `dreams/results/VALIDATION_SUMMARY.md`.
4. Confirm the current public version map is still correct in `dreams`.

## Final release steps

1. Commit the release prep.
2. Create the tag.
3. Push branch and tag.
4. Open or merge the release PR.
5. Verify GitHub release notes and the companion `dreams` surface after publication.
