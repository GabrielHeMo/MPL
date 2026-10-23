"""Structured status messages shared by MPL custom optimizers.

Messages are written to stdout using the same ``__MPL__`` prefix as the
JSONL evaluation bridge. The C# bridge consumes these status records while it
also services Aspen ``evaluate_batch`` requests from Python.
"""

from __future__ import annotations

import json
from typing import Any, Optional

PROTOCOL_PREFIX = "__MPL__"


def emit_status(
    message: str,
    *,
    level: str = "INFO",
    progress: Optional[float] = None,
    stage: Optional[str] = None,
    current: Optional[int] = None,
    total: Optional[int] = None,
    log: bool = True,
    algorithm: Optional[str] = None,
    **extra: Any,
) -> None:
    """Emit one structured MPL status record.

    ``log=False`` updates the UI status/progress without adding a permanent
    Activity Log entry. This is useful for high-frequency progress updates.
    """

    payload: dict[str, Any] = {
        "type": "status",
        "level": str(level or "INFO").upper(),
        "message": str(message),
        "log": bool(log),
    }

    if progress is not None:
        payload["progress"] = max(0.0, min(100.0, float(progress)))
    if stage:
        payload["stage"] = str(stage)
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)
    if algorithm:
        payload["algorithm"] = str(algorithm)

    for key, value in extra.items():
        if value is not None:
            payload[key] = value

    print(
        PROTOCOL_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
