"""CLI for summarizing recent tropical-mcp telemetry."""

from __future__ import annotations

import argparse
import json

from .server import telemetry_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize recent tropical-mcp telemetry records.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    result = telemetry_summary(limit=args.limit, run_id=args.run_id)
    if "error" in result:
        raise SystemExit(result["error"])

    print(json.dumps(result, indent=2))
