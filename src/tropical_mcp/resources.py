"""Packaged resource helpers for tropical-mcp."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

_PACKAGE_ROOT = files("tropical_mcp")
_FIXTURE_ROOT = _PACKAGE_ROOT.joinpath("fixtures")


def fixture_json(name: str) -> Any:
    """Load a packaged fixture as JSON."""
    return json.loads(_FIXTURE_ROOT.joinpath(name).read_text(encoding="utf-8"))


def fixture_ref(name: str) -> str:
    """Return a public, package-stable fixture reference."""
    return f"package:fixtures/{name}"
