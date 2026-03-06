"""Compatibility entrypoints for the previous tropical-compactor command names."""

from __future__ import annotations

import sys


def tropical_compactor() -> None:
    _warn("tropical-compactor", "tropical-mcp")
    from .server import main

    main()


def tropical_compactor_replay() -> None:
    _warn("tropical-compactor-replay", "tropical-mcp-replay")
    from .benchmark_harness import main

    main()


def tropical_compactor_full_validate() -> None:
    _warn("tropical-compactor-full-validate", "tropical-mcp-full-validate")
    from .validation import main

    main()


def _warn(old_name: str, new_name: str) -> None:
    print(
        f"Deprecated alias '{old_name}' invoked; use '{new_name}' instead. "
        "This compatibility shim will be removed after one release cycle.",
        file=sys.stderr,
    )

