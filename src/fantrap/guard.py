"""SQL guard: parse with sqlglot (duckdb dialect) and return violation codes with messages.

Checks, in order: parse errors, multiple statements, non-SELECT statements, a deny-list of
functions, table access (unknown and role-forbidden), column access (unknown and role-forbidden,
after qualifying columns and expanding `SELECT *` through CTEs and subqueries), and fan/chasm traps.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from fnmatch import fnmatch

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from .catalog import Catalog
from .traps import find_traps

# sqlglot logs a warning for every statement it can only parse as a raw command (LOAD, EXPLAIN).
logging.getLogger("sqlglot").setLevel(logging.ERROR)

PARSE_ERROR = "PARSE_ERROR"
MULTI_STATEMENT = "MULTI_STATEMENT"
WRITE_STATEMENT = "WRITE_STATEMENT"
FORBIDDEN_FUNCTION = "FORBIDDEN_FUNCTION"
FORBIDDEN_TABLE = "FORBIDDEN_TABLE"
FORBIDDEN_COLUMN = "FORBIDDEN_COLUMN"
UNKNOWN_TABLE = "UNKNOWN_TABLE"
UNKNOWN_COLUMN = "UNKNOWN_COLUMN"
FAN_TRAP = "FAN_TRAP"
CHASM_TRAP = "CHASM_TRAP"

ALL_CODES = (
    PARSE_ERROR,
    MULTI_STATEMENT,
    WRITE_STATEMENT,
    FORBIDDEN_FUNCTION,
    FORBIDDEN_TABLE,
    FORBIDDEN_COLUMN,
    UNKNOWN_TABLE,
    UNKNOWN_COLUMN,
    FAN_TRAP,
    CHASM_TRAP,
)

# Functions that read files, the process environment or run SQL from a string.
FORBIDDEN_FUNCTION_PATTERNS = (
    "read_csv*",
    "read_parquet*",
    "read_json*",
    "read_text",
    "read_blob",
    "parquet_scan",
    "glob",
    "getenv",
    "query",
    "query_table",
)
# Table functions that only generate rows.
ALLOWED_TABLE_FUNCTIONS = ("unnest", "generate_series", "range")

_WRITE_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Copy,
    exp.Attach,
    exp.Pragma,
    exp.Set,
    exp.Command,
    exp.Install,
    exp.TruncateTable,
)
_FILE_LIKE = re.compile(r"[/\\]|\.(csv|parquet|json|ndjson|db|duckdb)$", re.IGNORECASE)


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    suggestion: str | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def violation_codes(violations: list[Violation]) -> list[str]:
    """Distinct codes in first-seen order."""
    return list(dict.fromkeys(v.code for v in violations))


def _suggest(name: str, candidates: list[str]) -> str | None:
    pool = sorted(set(candidates))
    close = difflib.get_close_matches(name.lower(), pool, n=1, cutoff=0.5)
    if close:
        return close[0]
    contained = [c for c in pool if len(c) >= 4 and (c in name.lower() or name.lower() in c)]
    return contained[0] if contained else None


def _parse_message(exc: SqlglotError) -> str:
    errors = getattr(exc, "errors", None)
    if errors:
        first = errors[0]
        where = f" (line {first['line']}, column {first['col']})" if first.get("line") else ""
        return f"{first.get('description', 'Syntax error')}{where}"[:300]
    return re.sub(r"\x1b\[[0-9;]*m", "", " ".join(str(exc).split()))[:300]


def _is_alias_reference(col: exp.Column, name: str) -> bool:
    """True when a bare column name refers to a select alias rather than to a source column.

    A column inside its own projection (`total_amount AS total_amount`) is a source reference.
    """
    projection = col.find_ancestor(exp.Alias)
    if projection is not None and isinstance(projection.parent, exp.Select):
        return projection.alias.lower() != name
    return True


def _did_you_mean(kind: str, name: str, suggestion: str | None) -> str:
    base = f"Unknown {kind} '{name}'"
    return f"{base}. Did you mean '{suggestion}'?" if suggestion else f"{base}."


class Guard:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def check(self, sql: str, role: str) -> list[Violation]:
        self.catalog.role(role)  # unknown roles are a caller error, not a SQL violation

        statements, error = self._parse(sql)
        if error is not None:
            return [Violation(PARSE_ERROR, error)]
        if len(statements) > 1:
            return [
                Violation(
                    MULTI_STATEMENT,
                    f"{len(statements)} statements found; send exactly one SELECT statement.",
                )
            ]
        stmt = statements[0]
        if not isinstance(stmt, exp.Query):
            return [Violation(WRITE_STATEMENT, self._statement_message(stmt))]

        violations: list[Violation] = []
        nested = [n for n in stmt.walk() if isinstance(n, _WRITE_NODES)]
        if nested:
            kind = type(nested[0]).__name__.upper()
            violations.append(Violation(WRITE_STATEMENT, f"{kind} inside a query is not allowed."))

        violations.extend(self._functions(stmt))
        violations.extend(self._tables(stmt, role))
        violations.extend(self._opaque_column_access(stmt, role))
        if any(v.code == UNKNOWN_TABLE for v in violations):
            return violations

        qualified, qualify_violations = self._qualify(stmt)
        violations.extend(qualify_violations)
        if qualified is None:
            return violations
        column_violations = self._columns(qualified, role)
        column_violations.extend(self._whole_row_references(qualified, role))
        violations.extend(column_violations)
        if not any(v.code in (UNKNOWN_COLUMN, UNKNOWN_TABLE) for v in column_violations):
            for finding in find_traps(qualified, self.catalog):
                violations.append(Violation(finding.code, finding.message))
        return violations

    # ---- parsing -------------------------------------------------------------------------

    @staticmethod
    def _parse(sql: str) -> tuple[list[exp.Expression], str | None]:
        try:
            parsed = sqlglot.parse(sql, read="duckdb")
        except SqlglotError as exc:
            return [], _parse_message(exc)
        statements = [s for s in parsed if s is not None]
        if not statements:
            return [], "Empty statement."
        return statements, None

    @staticmethod
    def _statement_message(stmt: exp.Expression) -> str:
        kind = stmt.name.upper() if isinstance(stmt, exp.Command) else type(stmt).__name__.upper()
        return f"{kind} statements are not allowed; only SELECT queries may run."

    # ---- functions -----------------------------------------------------------------------

    @staticmethod
    def _function_name(node: exp.Expression) -> str | None:
        if isinstance(node, exp.Anonymous):
            return str(node.name).lower()
        if isinstance(node, exp.Func):
            return node.sql_name().lower()
        return None

    def _functions(self, stmt: exp.Expression) -> list[Violation]:
        found: list[str] = []
        for node in stmt.walk():
            name = self._function_name(node)
            if name and any(fnmatch(name, pat) for pat in FORBIDDEN_FUNCTION_PATTERNS):
                found.append(name)
            if isinstance(node, exp.Table) and not isinstance(node.this, exp.Identifier):
                table_fn = self._function_name(node.this)
                if table_fn and table_fn not in ALLOWED_TABLE_FUNCTIONS and table_fn not in found:
                    found.append(table_fn)
        return [
            Violation(FORBIDDEN_FUNCTION, f"Function {name}() is not allowed.")
            for name in dict.fromkeys(found)
        ]

    # ---- tables --------------------------------------------------------------------------

    def _tables(self, stmt: exp.Expression, role: str) -> list[Violation]:
        known = self.catalog.table_names()
        allowed = set(self.catalog.allowed_tables(role))
        out: list[Violation] = []
        seen: set[tuple[str, str]] = set()
        for scope in traverse_scope(stmt):
            for _, src in scope.selected_sources.values():
                if not isinstance(src, exp.Table) or not isinstance(src.this, exp.Identifier):
                    continue
                name = src.name.lower()
                if _FILE_LIKE.search(name):
                    if (FORBIDDEN_FUNCTION, name) not in seen:
                        seen.add((FORBIDDEN_FUNCTION, name))
                        out.append(
                            Violation(
                                FORBIDDEN_FUNCTION, f"Reading the file '{src.name}' is not allowed."
                            )
                        )
                    continue
                if src.db and src.db.lower() != "main":
                    code, message, suggestion = (
                        UNKNOWN_TABLE,
                        f"Schema-qualified table '{src.db}.{src.name}' is not in the catalog.",
                        None,
                    )
                elif name not in known:
                    suggestion = _suggest(name, known)
                    code = UNKNOWN_TABLE
                    message = _did_you_mean("table", src.name, suggestion)
                elif name not in allowed:
                    code, suggestion = FORBIDDEN_TABLE, None
                    message = f"Role '{role}' may not read table '{name}'."
                else:
                    continue
                if (code, name) not in seen:
                    seen.add((code, name))
                    out.append(Violation(code, message, suggestion))
        return out

    # ---- columns -------------------------------------------------------------------------

    def _opaque_column_access(self, stmt: exp.Expression, role: str) -> list[Violation]:
        """Constructs that read columns without naming them, which the allow-list cannot check.

        They are rejected outright for any role that has denied columns: COLUMNS(...) patterns,
        positional references (#3) and table aliases that rename columns (t(a, b, c)).
        """
        if not self.catalog.denied_columns(role):
            return []
        out: list[Violation] = []
        reasons = {
            exp.Columns: "COLUMNS(...) selects columns by pattern",
            exp.PositionalColumn: "positional column references (#n) hide which column is read",
        }
        for node in stmt.walk():
            for node_type, why in reasons.items():
                if isinstance(node, node_type):
                    out.append(
                        Violation(
                            FORBIDDEN_COLUMN,
                            f"Role '{role}' has restricted columns, so {why}. Name the columns.",
                        )
                    )
            if isinstance(node, exp.TableAlias) and node.args.get("columns"):
                out.append(
                    Violation(
                        FORBIDDEN_COLUMN,
                        f"Role '{role}' has restricted columns, so renaming columns in an "
                        "alias list is not allowed. Name the columns.",
                    )
                )
        return list({v.message: v for v in out}.values())

    def _whole_row_references(self, qualified: exp.Expression, role: str) -> list[Violation]:
        """A bare table alias used as a value (`SELECT c FROM customers c`) is the whole row,
        including restricted columns."""
        denied = self.catalog.denied_columns(role)
        denied_tables = {table for table, _ in denied}
        if not denied:
            return []
        out: list[Violation] = []
        if any(isinstance(c.this, exp.Star) for c in qualified.find_all(exp.Column)):
            out.append(
                Violation(
                    FORBIDDEN_COLUMN,
                    f"Role '{role}' has restricted columns, and `t.*` inside an expression "
                    "cannot be checked against them. List the columns you need.",
                )
            )
        seen: set[str] = set()
        for scope in traverse_scope(qualified):
            available = {
                c for _, src in scope.selected_sources.values() for c in self._source_columns(src)
            }
            for col in scope.expression.find_all(exp.TableColumn):
                if col.find_ancestor(exp.Select) is not scope.expression:
                    continue
                name = col.name.lower()
                src = self._resolve(scope, name)
                if (
                    isinstance(src, exp.Table)
                    and src.name.lower() in denied_tables
                    and name not in available
                    and name not in seen
                ):
                    seen.add(name)
                    out.append(
                        Violation(
                            FORBIDDEN_COLUMN,
                            f"Role '{role}' may not read whole rows of '{src.name.lower()}' "
                            f"(`{col.name}` as a value includes restricted columns). "
                            "List the columns you need.",
                        )
                    )
        return out

    def _qualify(self, stmt: exp.Expression) -> tuple[exp.Expression | None, list[Violation]]:
        try:
            qualified = qualify(
                stmt.copy(),
                schema=self.catalog.schema(),
                dialect="duckdb",
                validate_qualify_columns=False,
                identify=False,
            )
        except OptimizeError as exc:
            message = str(exc)
            match = re.search(r"Unknown column: (\S+)", message)
            if match:
                return None, [self._unknown_qualified_column(stmt, match.group(1))]
            return None, [Violation(PARSE_ERROR, f"Could not analyse the query: {message}"[:300])]
        return qualified, []

    def _unknown_qualified_column(self, stmt: exp.Expression, name: str) -> Violation:
        candidates: list[str] = []
        for scope in traverse_scope(stmt):
            for col in scope.columns:
                if col.name.lower() == name.lower() and col.table:
                    src = scope.sources.get(col.table)
                    candidates.extend(self._source_columns(src))
        if not candidates:
            candidates = [c for t in self.catalog.table_names() for c in self.catalog.columns(t)]
        suggestion = _suggest(name, candidates)
        return Violation(UNKNOWN_COLUMN, _did_you_mean("column", name, suggestion), suggestion)

    def _source_columns(self, src: exp.Table | Scope | None) -> list[str]:
        if isinstance(src, exp.Table):
            name = src.name.lower()
            return self.catalog.columns(name) if name in self.catalog.tables else []
        if isinstance(src, Scope):
            return [str(n).lower() for n in src.expression.named_selects]
        return []

    @staticmethod
    def _resolve(scope: Scope, alias: str) -> exp.Table | Scope | None:
        current: Scope | None = scope
        while current is not None:
            if alias in current.sources:
                return current.sources[alias]
            current = current.parent
        return None

    def _columns(self, qualified: exp.Expression, role: str) -> list[Violation]:
        denied = self.catalog.denied_columns(role)
        out: list[Violation] = []
        seen: set[tuple[str, str]] = set()
        for scope in traverse_scope(qualified):
            aliases = {str(n).lower() for n in scope.expression.named_selects}
            for col in [*scope.columns, *scope.external_columns]:
                name = col.name.lower()
                if col.table:
                    src = self._resolve(scope, col.table)
                    if src is None:
                        key = (UNKNOWN_TABLE, col.table)
                        if key not in seen:
                            seen.add(key)
                            out.append(
                                Violation(
                                    UNKNOWN_TABLE,
                                    f"Unknown table or alias '{col.table}' in {col.sql('duckdb')}.",
                                )
                            )
                    elif isinstance(src, exp.Table) and (src.name.lower(), name) in denied:
                        key = (FORBIDDEN_COLUMN, f"{src.name.lower()}.{name}")
                        if key not in seen:
                            seen.add(key)
                            out.append(
                                Violation(
                                    FORBIDDEN_COLUMN,
                                    f"Role '{role}' may not read column '{key[1]}'. "
                                    "List the columns you need instead of selecting *.",
                                )
                            )
                    continue
                if name in aliases and _is_alias_reference(col, name):
                    continue
                available: list[str] = []
                for src in scope.selected_sources.values():
                    available.extend(self._source_columns(src[1]))
                if name in available:
                    continue  # ambiguous between sources; the engine reports it
                if isinstance(col.this, exp.Star):
                    continue
                key = (UNKNOWN_COLUMN, name)
                if key not in seen:
                    seen.add(key)
                    suggestion = _suggest(name, available or self._all_columns())
                    out.append(
                        Violation(
                            UNKNOWN_COLUMN,
                            _did_you_mean("column", col.name, suggestion),
                            suggestion,
                        )
                    )
        return out

    def _all_columns(self) -> list[str]:
        return [c for t in self.catalog.table_names() for c in self.catalog.columns(t)]
