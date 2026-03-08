from __future__ import annotations

from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _expected_files(base: Path) -> list[str]:
    return sorted(
        path.relative_to(ROOT).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )


def _manifest_entries(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        entries.append((digest, rel))
    return entries


def test_public_release_docs_exist() -> None:
    expected = [
        ROOT / "docs" / "ARCHITECTURE.md",
        ROOT / "docs" / "METHODOLOGY.md",
        ROOT / "docs" / "ARTIFACT_INDEX.md",
        ROOT / "docs" / "SOURCE_OF_TRUTH.md",
        ROOT / "scripts" / "update_release_checksums.sh",
    ]

    for path in expected:
        assert path.exists(), f"missing release-facing file: {path}"


def test_checksum_manifests_cover_examples_and_fixtures() -> None:
    manifests = [
        ROOT / "examples" / "codex" / "SHA256SUMS.txt",
        ROOT / "src" / "tropical_mcp" / "fixtures" / "SHA256SUMS.txt",
    ]

    for manifest in manifests:
        entries = _manifest_entries(manifest)
        assert entries, f"empty checksum manifest: {manifest}"

        seen_files = [rel for _, rel in entries]
        assert seen_files == _expected_files(manifest.parent)

        for digest, rel in entries:
            target = ROOT / rel
            assert target.is_file(), f"missing manifest target: {rel}"
            assert sha256(target.read_bytes()).hexdigest() == digest


def test_release_docs_reference_integrity_flow() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    release_doc = (ROOT / "docs" / "RELEASE.md").read_text(encoding="utf-8")
    security_doc = (ROOT / "SECURITY.md").read_text(encoding="utf-8")

    assert "minimum smoke sequence" in readme
    assert "fuller research review" in readme
    assert "./scripts/update_release_checksums.sh" in release_doc
    assert "Do not open public issues for unpatched vulnerabilities." in security_doc
