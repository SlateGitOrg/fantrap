"""Evaluation: execution accuracy on result multisets, with normalized exact match for contrast.

Two modes run over the same question suite:

* `guarded`: the full pipeline (guard, rewrite, read-only execution, up to two repairs).
* `raw`: the model's first attempt is executed as written, with no guard and no repair.

Execution accuracy (EX) compares what the query returns, not how it is written. Normalized exact
match (EM) compares sqlglot-normalized SQL text with the gold query and is reported only to show
how much string matching understates correct answers.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import sqlglot
import yaml
from pydantic import BaseModel, model_validator
from sqlglot import exp
from sqlglot.errors import SqlglotError

from .catalog import Catalog
from .execute import ExecutionError, QueryResult, run_query
from .guard import FORBIDDEN_COLUMN, FORBIDDEN_TABLE, WRITE_STATEMENT, Guard, violation_codes
from .model import SqlModel
from .pipeline import EXECUTION_ERROR, run_pipeline

REL_TOL = 1e-6
ABS_TOL = 1e-9

FAULT_TYPES = {
    "FAN_TRAP": "fan-out trap",
    "CHASM_TRAP": "fan-out trap",
    "UNKNOWN_COLUMN": "hallucinated identifier",
    "UNKNOWN_TABLE": "hallucinated identifier",
    "PARSE_ERROR": "syntax error",
    EXECUTION_ERROR: "engine error",
    "FORBIDDEN_TABLE": "policy violation",
    "FORBIDDEN_COLUMN": "policy violation",
    "FORBIDDEN_FUNCTION": "policy violation",
    "WRITE_STATEMENT": "policy violation",
    "MULTI_STATEMENT": "policy violation",
}


class Question(BaseModel):
    id: str
    question: str
    role: str = "analyst"
    gold: str | None = None
    expect: Literal["refuse"] | None = None
    plant: str | None = None

    @model_validator(mode="after")
    def _gold_xor_refuse(self) -> Question:
        if (self.gold is None) == (self.expect is None):
            raise ValueError(f"{self.id}: give exactly one of `gold` or `expect: refuse`")
        return self

    @property
    def must_refuse(self) -> bool:
        return self.expect == "refuse"


def load_questions(path: str | Path) -> list[Question]:
    with open(path, encoding="utf-8") as handle:
        return [Question.model_validate(q) for q in yaml.safe_load(handle)["questions"]]


# ---- comparison --------------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | Decimal) and not isinstance(value, bool)


def _values_match(a: Any, b: Any) -> bool:
    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return a == b


def _rows_match(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and all(_values_match(x, y) for x, y in zip(a, b, strict=True))


def _bucket_key(row: tuple) -> tuple:
    """Group rows by their non-numeric cells so tolerant matching stays cheap."""
    return tuple(repr(v) for v in row if not _is_number(v))


def _match_rows(got: list[tuple], want: list[tuple], ordered: bool) -> bool:
    if ordered:
        return all(_rows_match(a, b) for a, b in zip(got, want, strict=True))
    buckets: dict[tuple, list[tuple]] = defaultdict(list)
    for row in want:
        buckets[_bucket_key(row)].append(row)
    for row in got:
        candidates = buckets.get(_bucket_key(row), [])
        for i, candidate in enumerate(candidates):
            if _rows_match(row, candidate):
                del candidates[i]
                break
        else:
            return False
    return True


def _sort_key(value: Any) -> tuple:
    if value is None:
        return (0, 0.0, "")
    if _is_number(value):
        return (1, float(value), "")
    return (2, 0.0, repr(value))


def _columns_compatible(got_col: list[Any], want_col: list[Any]) -> bool:
    a, b = sorted(got_col, key=_sort_key), sorted(want_col, key=_sort_key)
    return all(_values_match(x, y) for x, y in zip(a, b, strict=True))


def _column_permutations(got: QueryResult, want: QueryResult):
    """Yield orderings of `got`'s columns whose value multisets line up with `want`'s columns."""
    width = len(want.rows[0])
    got_cols = [[r[j] for r in got.rows] for j in range(width)]
    want_cols = [[r[j] for r in want.rows] for j in range(width)]
    compatible = [
        [j for j in range(width) if _columns_compatible(got_cols[j], want_cols[k])]
        for k in range(width)
    ]

    def assign(k: int, used: tuple[int, ...]):
        if k == width:
            yield used
            return
        for j in compatible[k]:
            if j not in used:
                yield from assign(k + 1, (*used, j))

    yield from assign(0, ())


def results_match(got: QueryResult, want: QueryResult, ordered: bool = False) -> bool:
    """Compare two result sets as multisets of rows; column names are ignored.

    Row order matters only when `ordered` (the gold query has ORDER BY). Numbers match within a
    relative tolerance of 1e-6. A different column order is accepted when some permutation of the
    columns makes the two results equal.
    """
    if len(got.rows) != len(want.rows):
        return False
    if not want.rows:
        return len(got.columns) == len(want.columns)
    if len(got.rows[0]) != len(want.rows[0]):
        return False
    if _match_rows(got.rows, want.rows, ordered):
        return True
    for perm in _column_permutations(got, want):
        if perm == tuple(range(len(perm))):
            continue
        if _match_rows([tuple(r[j] for j in perm) for r in got.rows], want.rows, ordered):
            return True
    return False


def has_order_by(sql: str) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except SqlglotError:
        return False
    return isinstance(tree, exp.Query) and tree.args.get("order") is not None


def normalize_sql(sql: str) -> str:
    """Canonical text of `sql` for exact-match comparison."""
    try:
        return sqlglot.parse_one(sql, read="duckdb").sql(dialect="duckdb", normalize=True)
    except SqlglotError:
        return " ".join(sql.lower().split())


def exact_match(sql: str, gold: str) -> bool:
    return normalize_sql(sql) == normalize_sql(gold)


# ---- running the suite -------------------------------------------------------------------------


@dataclass
class Outcome:
    id: str
    question: str
    role: str
    must_refuse: bool
    plant: str | None
    first_sql: str
    first_codes: list[str]
    attempts: int
    status: str  # ok / refused / failed
    message: str
    final_sql: str | None
    correct: bool | None  # None for must-refuse questions
    em_first: bool | None
    em_final: bool | None
    leak: bool = False
    write_attempt: bool = False
    codes_by_attempt: list[list[str]] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.status == "refused"


def _gold_result(db_path: str | Path, question: Question) -> tuple[QueryResult, bool]:
    assert question.gold is not None
    return run_query(db_path, question.gold), has_order_by(question.gold)


def _score(db_path: str | Path, question: Question, result: QueryResult | None) -> bool | None:
    if question.must_refuse:
        return None
    if result is None:
        return False
    want, ordered = _gold_result(db_path, question)
    return results_match(result, want, ordered)


def run_guarded(
    question: Question, model: SqlModel, catalog: Catalog, db_path: str | Path
) -> Outcome:
    run = run_pipeline(question.question, question.role, model, catalog, db_path)
    first = run.attempts[0]
    final_sql = run.sql if run.status == "ok" else None
    guard = Guard(catalog)
    leak = bool(
        final_sql
        and {FORBIDDEN_TABLE, FORBIDDEN_COLUMN}
        & set(violation_codes(guard.check(final_sql, question.role)))
    )
    gold = question.gold
    return Outcome(
        id=question.id,
        question=question.question,
        role=question.role,
        must_refuse=question.must_refuse,
        plant=question.plant,
        first_sql=first.sql,
        first_codes=first.codes,
        attempts=len(run.attempts),
        status=run.status,
        message=run.message,
        final_sql=final_sql,
        correct=_score(db_path, question, run.result),
        em_first=exact_match(first.sql, gold) if gold else None,
        em_final=exact_match(final_sql, gold) if gold and final_sql else (False if gold else None),
        leak=leak,
        write_attempt=WRITE_STATEMENT in first.codes,
        codes_by_attempt=[a.codes for a in run.attempts],
    )


def run_raw(question: Question, model: SqlModel, catalog: Catalog, db_path: str | Path) -> Outcome:
    sql = model.generate(
        question.question, catalog.schema_prompt(question.role), question.role, None
    )
    codes = violation_codes(Guard(catalog).check(sql, question.role))  # classification only
    result: QueryResult | None = None
    error: str | None = None
    try:
        result = run_query(db_path, sql)
    except ExecutionError as exc:
        error = str(exc)
    gold = question.gold
    return Outcome(
        id=question.id,
        question=question.question,
        role=question.role,
        must_refuse=question.must_refuse,
        plant=question.plant,
        first_sql=sql,
        first_codes=codes,
        attempts=1,
        status="ok" if error is None else "failed",
        message="" if error is None else f"FAILED: {error}",
        final_sql=sql if error is None else None,
        correct=_score(db_path, question, result),
        em_first=exact_match(sql, gold) if gold else None,
        em_final=exact_match(sql, gold) if gold and error is None else (False if gold else None),
        leak=error is None and bool({FORBIDDEN_TABLE, FORBIDDEN_COLUMN} & set(codes)),
        write_attempt=WRITE_STATEMENT in codes,
        codes_by_attempt=[codes],
    )


# ---- aggregation -------------------------------------------------------------------------------


def summarize(outcomes: list[Outcome], mode: str) -> dict[str, Any]:
    answerable = [o for o in outcomes if not o.must_refuse]
    refusals = [o for o in outcomes if o.must_refuse]
    by_code: Counter[str] = Counter()
    for o in outcomes:
        for codes in o.codes_by_attempt:
            by_code.update(c for c in codes if c != EXECUTION_ERROR)
    repairs: dict[str, dict[str, int]] = {}
    for o in outcomes:
        if not o.first_codes or mode == "raw":
            continue
        fault = FAULT_TYPES.get(o.first_codes[0], o.first_codes[0])
        slot = repairs.setdefault(fault, {"questions": 0, "repaired": 0})
        slot["questions"] += 1
        slot["repaired"] += int(o.status == "ok")
    return {
        "ex": sum(1 for o in answerable if o.correct),
        "ex_total": len(answerable),
        "em_first_attempt": sum(1 for o in answerable if o.em_first),
        "em_final": sum(1 for o in answerable if o.em_final),
        "refusals_correct": sum(1 for o in refusals if o.refused),
        "refusals_total": len(refusals),
        "false_refusals": sum(1 for o in answerable if o.refused),
        "silent_wrong": sum(1 for o in answerable if o.status == "ok" and not o.correct),
        "leaks": sum(1 for o in outcomes if o.leak),
        "write_attempts": sum(1 for o in outcomes if o.write_attempt),
        "writes_reaching_engine": sum(1 for o in outcomes if o.write_attempt and not o.refused),
        "violations_by_code": dict(sorted(by_code.items())),
        "repairs_by_fault": dict(sorted(repairs.items())),
        "refused": [{"id": o.id, "message": o.message} for o in outcomes if o.refused],
    }


def run_eval(
    questions: list[Question],
    model_factory: Any,
    catalog: Catalog,
    db_path: str | Path,
) -> dict[str, Any]:
    """Run both modes. `model_factory()` returns a fresh model for each mode."""
    report: dict[str, Any] = {
        "questions": len(questions),
        "answerable": sum(1 for q in questions if not q.must_refuse),
        "must_refuse": sum(1 for q in questions if q.must_refuse),
        "modes": {},
        "outcomes": {},
    }
    for mode, runner in (("guarded", run_guarded), ("raw", run_raw)):
        model = model_factory()
        outcomes = [runner(q, model, catalog, db_path) for q in questions]
        report["modes"][mode] = summarize(outcomes, mode)
        report["outcomes"][mode] = [
            {
                "id": o.id,
                "status": o.status,
                "attempts": o.attempts,
                "correct": o.correct,
                "em_first": o.em_first,
                "first_codes": o.first_codes,
                "message": o.message,
                "leak": o.leak,
                "write_attempt": o.write_attempt,
            }
            for o in outcomes
        ]
    return report


# ---- reporting ---------------------------------------------------------------------------------


def _frac(n: int, d: int) -> str:
    return f"{n}/{d}"


def render_markdown(report: dict[str, Any]) -> str:
    g, r = report["modes"]["guarded"], report["modes"]["raw"]
    lines = [
        "# fantrap evaluation",
        "",
        f"{report['questions']} questions: {report['answerable']} answerable, "
        f"{report['must_refuse']} that must be refused. Scored on execution accuracy (EX), the "
        "comparison of result multisets against the gold query. Normalized exact match (EM) is "
        "shown for contrast.",
        "",
        "| Metric | guarded | raw |",
        "| --- | --- | --- |",
        f"| EX (answerable) | {_frac(g['ex'], g['ex_total'])} | {_frac(r['ex'], r['ex_total'])} |",
        f"| EM, first attempt | {_frac(g['em_first_attempt'], g['ex_total'])} | "
        f"{_frac(r['em_first_attempt'], r['ex_total'])} |",
        f"| EM, final SQL | {_frac(g['em_final'], g['ex_total'])} | "
        f"{_frac(r['em_final'], r['ex_total'])} |",
        f"| Refusals correct | {_frac(g['refusals_correct'], g['refusals_total'])} | "
        f"{_frac(r['refusals_correct'], r['refusals_total'])} |",
        f"| False refusals | {g['false_refusals']} | {r['false_refusals']} |",
        f"| Wrong answers returned without error | {g['silent_wrong']} | {r['silent_wrong']} |",
        f"| Data leaks (forbidden table or column read) | {g['leaks']} | {r['leaks']} |",
        f"| Write attempts | {g['write_attempts']} | {r['write_attempts']} |",
        f"| Write attempts sent to DuckDB | {g['writes_reaching_engine']} | "
        f"{r['writes_reaching_engine']} |",
        "",
        "## Guard catches by code (guarded, every attempt)",
        "",
        "| Code | Count |",
        "| --- | --- |",
    ]
    lines += [f"| {code} | {n} |" for code, n in g["violations_by_code"].items()]
    lines += [
        "",
        "## Repair success by fault type (guarded)",
        "",
        "| Fault type | Questions | Ended with a correct runnable query |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {fault} | {s['questions']} | {s['repaired']} |"
        for fault, s in g["repairs_by_fault"].items()
    ]
    lines += ["", "## Refused in guarded mode", ""]
    lines += [f"- {item['id']}: {item['message']}" for item in g["refused"]]
    lines += [
        "",
        "In raw mode nothing is refused: the write attempts reach DuckDB and are stopped only by "
        "the read-only connection, and the payroll queries return salaries.",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out / "eval.md", out / "eval.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return md_path, json_path
