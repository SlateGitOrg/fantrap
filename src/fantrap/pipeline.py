"""generate -> guard -> rewrite -> execute, with up to two repair rounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .catalog import Catalog
from .execute import ExecutionError, QueryResult, run_query
from .guard import Guard, Violation, violation_codes
from .model import SqlModel
from .rewrite import inject_limit

MAX_REPAIRS = 2
EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass
class Attempt:
    number: int
    sql: str
    violations: list[Violation] = field(default_factory=list)
    execution_error: str | None = None

    @property
    def codes(self) -> list[str]:
        codes = violation_codes(self.violations)
        if self.execution_error is not None:
            codes.append(EXECUTION_ERROR)
        return codes

    @property
    def ok(self) -> bool:
        return not self.violations and self.execution_error is None


@dataclass
class PipelineResult:
    status: str  # "ok", "refused" or "failed"
    attempts: list[Attempt]
    sql: str | None = None
    result: QueryResult | None = None
    message: str = ""

    @property
    def refused(self) -> bool:
        return self.status == "refused"

    @property
    def repaired(self) -> bool:
        return self.status == "ok" and len(self.attempts) > 1


def build_feedback(attempt: Attempt) -> str:
    lines = ["Your previous SQL was rejected.", "Previous SQL:", attempt.sql, "Problems:"]
    lines.extend(f"- {v.code}: {v.message}" for v in attempt.violations)
    if attempt.execution_error:
        lines.append(f"- {EXECUTION_ERROR}: {attempt.execution_error}")
    lines.append("Return a corrected single SELECT statement.")
    return "\n".join(lines)


def run_pipeline(
    question: str,
    role: str,
    model: SqlModel,
    catalog: Catalog,
    db_path: str | Path,
    max_repairs: int = MAX_REPAIRS,
) -> PipelineResult:
    guard = Guard(catalog)
    schema_prompt = catalog.schema_prompt(role)
    attempts: list[Attempt] = []
    feedback: str | None = None
    for number in range(1, max_repairs + 2):
        sql = model.generate(question, schema_prompt, role, feedback)
        attempt = Attempt(number, sql, guard.check(sql, role))
        attempts.append(attempt)
        if not attempt.violations:
            try:
                result = run_query(db_path, inject_limit(sql))
            except ExecutionError as exc:
                attempt.execution_error = str(exc)
            else:
                return PipelineResult("ok", attempts, sql=sql, result=result)
        feedback = build_feedback(attempt)

    last = attempts[-1]
    if last.violations:
        codes = ", ".join(violation_codes(last.violations))
        return PipelineResult("refused", attempts, sql=last.sql, message=f"REFUSED: {codes}")
    return PipelineResult(
        "failed", attempts, sql=last.sql, message=f"FAILED: {last.execution_error}"
    )
