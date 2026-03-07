#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] Running unit and property tests"
uv run --extra dev pytest -q

echo "[2/6] Building distributable artifacts"
uv build

echo "[3/6] Verifying the built wheel"
./scripts/validate_installed_wheel.sh

echo "[4/6] Running comprehensive functional validation"
uv run tropical-mcp-full-validate

echo "[5/6] Running quick replay benchmark sweep"
uv run tropical-mcp-replay \
  --fractions 1.0,0.8,0.65,0.5,0.4 \
  --policies recency,l2_guarded \
  --k 3 \
  --line-count 200 \
  --output-dir artifacts/cyberops_mcp_replay_quickcheck

echo "[6/6] Validation complete"
echo "Artifacts:"
echo "  - artifacts/cyberops_mcp_replay_quickcheck/replay_rows.csv"
echo "  - artifacts/cyberops_mcp_replay_quickcheck/replay_summary.csv"
echo "  - artifacts/cyberops_mcp_replay_quickcheck/replay_summary.json"
