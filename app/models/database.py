"""
app/models/database.py
======================
Async SQLite audit log database using aiosqlite.
Stores every request/response decision for monitoring and retrospective analysis.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import aiosqlite

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    client_ip       TEXT    NOT NULL DEFAULT '',
    input_text      TEXT    NOT NULL DEFAULT '',
    input_verdict   TEXT    NOT NULL DEFAULT 'allow',
    input_confidence REAL   NOT NULL DEFAULT 0.0,
    input_threat    TEXT    NOT NULL DEFAULT 'none',
    output_verdict  TEXT    NOT NULL DEFAULT 'allow',
    output_threat   TEXT    NOT NULL DEFAULT 'none',
    latency_ms      REAL    NOT NULL DEFAULT 0.0
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log (timestamp);
"""


# ── Database helpers ──────────────────────────────────────────────────────────


def _db_path() -> str:
    """Resolve the SQLite file path, creating parent directories as needed."""
    settings = get_settings()
    # Strip the aiosqlite driver prefix so we have a plain file path
    raw = settings.database_url.replace("sqlite+aiosqlite:///", "")
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


async def init_db() -> None:
    """Create the audit_log table if it doesn't already exist."""
    db_path = _db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(CREATE_TABLE_SQL)
        await db.execute(CREATE_INDEX_SQL)
        await db.commit()
    logger.info("Audit database initialised at %s", db_path)


async def log_audit_entry(
    *,
    request_id: str,
    client_ip: str,
    input_text: str,
    input_verdict: str,
    input_confidence: float,
    input_threat: str,
    output_verdict: str,
    output_threat: str,
    latency_ms: float,
) -> None:
    """Insert one audit row. Fails silently to avoid impacting the request path."""
    settings = get_settings()
    # Optionally redact the prompt text before storing
    stored_input = "[REDACTED]" if settings.redact_prompts_in_logs else input_text

    db_path = _db_path()
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                INSERT INTO audit_log
                    (request_id, timestamp, client_ip,
                     input_text, input_verdict, input_confidence, input_threat,
                     output_verdict, output_threat, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    time.time(),
                    client_ip,
                    stored_input,
                    input_verdict,
                    input_confidence,
                    input_threat,
                    output_verdict,
                    output_threat,
                    latency_ms,
                ),
            )
            await db.commit()
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to write audit log entry: %s", exc)


async def get_recent_logs(limit: int = 100) -> list[dict]:
    """Fetch the most recent audit entries (for the admin/debug endpoint)."""
    db_path = _db_path()
    rows = []
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = [dict(row) for row in await cursor.fetchall()]
    return rows
