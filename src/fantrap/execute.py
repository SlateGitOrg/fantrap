"""Run SQL on a read-only DuckDB connection with a row cap.

This is the last line of defence: a write that slips past the guard still fails here, and file or
network access is switched off for the connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

MAX_ROWS = 10_000


class ExecutionError(Exception):
    """DuckDB rejected or failed the statement."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool = False


def connect_readonly(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    if not Path(db_path).exists():
        raise ExecutionError(f"database not found: {db_path} (run `fantrap seed` first)")
    try:
        return duckdb.connect(
            str(db_path), read_only=True, config={"enable_external_access": "false"}
        )
    except duckdb.Error as exc:
        raise ExecutionError(str(exc)) from exc


def run_query(db_path: str | Path, sql: str, max_rows: int = MAX_ROWS) -> QueryResult:
    """Execute `sql` read-only and fetch at most `max_rows` rows."""
    con = connect_readonly(db_path)
    try:
        cursor = con.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        fetched = cursor.fetchmany(max_rows + 1) if cursor.description else []
    except duckdb.Error as exc:
        raise ExecutionError(" ".join(str(exc).split())) from exc
    finally:
        con.close()
    return QueryResult(columns, [tuple(r) for r in fetched[:max_rows]], len(fetched) > max_rows)
