"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store a request and return its correlation ID."""
        request_id = request_id or uuid.uuid4().hex
        self._open[request_id] = {
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store the response and its decision, correlated to the input."""
        if request_id is None:
            request_id = next(
                (
                    key for key, item in reversed(list(self._open.items()))
                    if item["user_id"] == user_id
                ),
                None,
            )

        pending = self._open.pop(request_id, None) if request_id else None
        completed_at = utc_now_iso()
        started_monotonic = (
            pending["started_monotonic"] if pending else time.perf_counter()
        )
        entry = {
            "request_id": request_id or uuid.uuid4().hex,
            "user_id": pending["user_id"] if pending else user_id,
            "input": pending["input"] if pending else "",
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "started_at": pending["started_at"] if pending else completed_at,
            "completed_at": completed_at,
            "latency_seconds": max(0.0, time.perf_counter() - started_monotonic),
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
