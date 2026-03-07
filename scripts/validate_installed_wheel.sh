#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! compgen -G "dist/*.whl" > /dev/null; then
  echo "No wheel found in dist/. Run 'uv build' first." >&2
  exit 1
fi

wheel_path="$(ls -t dist/*.whl | head -n 1)"

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tropical-mcp-wheel-XXXXXX")"
cleanup() {
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

uv venv "$TEMP_DIR/.venv" > /dev/null 2>&1
uv pip install --python "$TEMP_DIR/.venv/bin/python" "$wheel_path" > /dev/null 2>&1
"$TEMP_DIR/.venv/bin/tropical-mcp-full-validate" > /dev/null
echo "Installed wheel validation passed for $(basename "$wheel_path")"
