"""Runtime settings and telemetry helpers for tropical-mcp."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

CLIENT_CHOICES = frozenset({"auto", "codex", "claude", "generic"})
SUPPORTED_POLICIES = ("l2_guarded", "l2_iterative_guarded", "recency")
SUPPORTED_TOOLS = (
    "compact",
    "inspect",
    "inspect_horizon",
    "compact_auto",
    "context_anchor",
    "certificate",
    "telemetry_summary",
    "retention_floor",
    "tag",
    "diagnose",
    "runtime_info",
)
TELEMETRY_SCHEMA_VERSION = 1

_CLAUDE_HINTS = ("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_PROJECT_DIR", "CLAUDE_CONFIG_DIR")
_PROCESS_RUN_ID = uuid.uuid4().hex


@dataclass(frozen=True)
class RuntimeSettings:
    client: str
    client_source: str
    telemetry_enabled: bool
    telemetry_path: str | None
    run_id: str | None
    package_version: str


def resolve_runtime_settings(env: Mapping[str, str] | None = None) -> RuntimeSettings:
    current_env = dict(os.environ if env is None else env)
    client, client_source = _resolve_client(current_env)

    telemetry_enabled = current_env.get("TROPICAL_MCP_ENABLE_TELEMETRY", "1") != "0"
    override_path = current_env.get("TROPICAL_MCP_TELEMETRY_PATH")
    telemetry_path = None
    if telemetry_enabled:
        path = (
            _expand_path(override_path, current_env)
            if override_path
            else _default_telemetry_path(client, current_env)
        )
        telemetry_path = str(path)

    run_id = current_env.get("TROPICAL_MCP_RUN_ID") or _PROCESS_RUN_ID
    package_version = _package_version()

    return RuntimeSettings(
        client=client,
        client_source=client_source,
        telemetry_enabled=telemetry_enabled,
        telemetry_path=telemetry_path,
        run_id=run_id,
        package_version=package_version,
    )


def runtime_info_payload(settings: RuntimeSettings) -> dict[str, object]:
    return {
        "client": settings.client,
        "client_source": settings.client_source,
        "telemetry_enabled": settings.telemetry_enabled,
        "telemetry_path": settings.telemetry_path,
        "run_id": settings.run_id,
        "package_version": settings.package_version,
        "supported_tools": list(SUPPORTED_TOOLS),
        "supported_policies": list(SUPPORTED_POLICIES),
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
    }


def _resolve_client(env: Mapping[str, str]) -> tuple[str, str]:
    raw_value = env.get("TROPICAL_MCP_CLIENT", "auto").strip().lower() or "auto"
    if raw_value in CLIENT_CHOICES and raw_value != "auto":
        return raw_value, "explicit"

    if raw_value == "auto":
        if env.get("CODEX_HOME"):
            return "codex", "auto"
        if any(env.get(key) for key in _CLAUDE_HINTS):
            return "claude", "auto"
        return "generic", "auto"

    return "generic", "auto"


def _default_telemetry_path(client: str, env: Mapping[str, str]) -> Path:
    if client == "codex":
        return _expand_path(env.get("CODEX_HOME") or "~/.codex", env) / "state" / "tropical-mcp" / "telemetry.jsonl"
    if client == "claude":
        return _expand_path("~/.claude/compactor-telemetry.jsonl", env)
    state_root = env.get("XDG_STATE_HOME") or "~/.local/state"
    return _expand_path(state_root, env) / "tropical-mcp" / "telemetry.jsonl"


def _expand_path(raw_path: str, env: Mapping[str, str]) -> Path:
    if raw_path.startswith("~"):
        home = env.get("HOME")
        if home:
            suffix = raw_path[2:] if raw_path.startswith("~/") else raw_path[1:]
            return Path(home).joinpath(suffix)
    return Path(raw_path).expanduser()


def _package_version() -> str:
    try:
        return version("tropical-mcp")
    except PackageNotFoundError:
        return "0.0.0+unknown"
