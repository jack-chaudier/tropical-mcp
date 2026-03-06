"""CLI for emitting memory safety certificate JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from .server import certificate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a tropical-mcp certificate JSON artifact.")
    parser.add_argument("--input", required=True, help="Path to a JSON file containing messages or {messages: [...]} payload.")
    parser.add_argument("--output", required=True, help="Where to write the certificate JSON.")
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--baseline-policy", default="recency")
    parser.add_argument("--guarded-policy", default="l2_guarded")
    parser.add_argument("--name", default="memory_safety_certificate")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    messages = _messages_from_payload(payload)
    result = certificate(
        messages=messages,
        token_budget=args.token_budget,
        k=args.k,
        baseline_policy=args.baseline_policy,
        guarded_policy=args.guarded_policy,
        name=args.name,
    )
    if "error" in result:
        raise SystemExit(result["error"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {output_path}")


def _messages_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return cast(list[dict[str, Any]], payload)
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        return cast(list[dict[str, Any]], payload["messages"])
    raise ValueError("input JSON must be a list of messages or an object with a 'messages' list")
