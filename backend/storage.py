from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.config import BASE_DIR, get_env
from backend.models import AnalyzeRequest, AuditReport, ReportRecord, ScrapedPage


DEFAULT_DB_PATH = BASE_DIR / "data" / "siteaudit.db"


def get_db_path() -> Path:
    configured = get_env("SITEAUDIT_DB_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_DB_PATH


def _connect() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    _ensure_schema(connection)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            request_json TEXT NOT NULL,
            source_json TEXT NOT NULL,
            report_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at DESC)"
    )
    connection.commit()


def init_db() -> None:
    with _connect():
        pass


def save_report(record: ReportRecord) -> None:
    with _connect() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO reports (
                id,
                created_at,
                request_json,
                source_json,
                report_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.created_at,
                record.request.model_dump_json(),
                record.source.model_dump_json(),
                record.report.model_dump_json(),
            ),
        )
        connection.commit()


def _row_to_record(row: sqlite3.Row) -> ReportRecord:
    return ReportRecord(
        id=row["id"],
        created_at=row["created_at"],
        request=AnalyzeRequest.model_validate_json(row["request_json"]),
        source=ScrapedPage.model_validate_json(row["source_json"]),
        report=AuditReport.model_validate_json(row["report_json"]),
    )


def get_report(report_id: str) -> ReportRecord | None:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, created_at, request_json, source_json, report_json
            FROM reports
            WHERE id = ?
            """,
            (report_id,),
        ).fetchone()

    if row is None:
        return None
    return _row_to_record(row)


def list_reports() -> list[ReportRecord]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, request_json, source_json, report_json
            FROM reports
            ORDER BY created_at DESC
            """
        ).fetchall()

    return [_row_to_record(row) for row in rows]


def clear_reports() -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM reports")
        connection.commit()


def report_exists(report_id: str) -> bool:
    with _connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM reports WHERE id = ? LIMIT 1",
            (report_id,),
        ).fetchone()
    return row is not None
