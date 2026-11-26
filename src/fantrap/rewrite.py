"""Safe rewrites applied to guarded SQL before it runs."""

from __future__ import annotations

import sqlglot
from sqlglot import exp

DEFAULT_LIMIT = 1000


def _is_aggregate_query(select: exp.Select) -> bool:
    """True for queries that return a summary: GROUP BY, or an aggregate outside a window."""
    if select.args.get("group") is not None:
        return True
    for projection in select.expressions:
        for agg in projection.find_all(exp.AggFunc):
            if agg.find_ancestor(exp.Window) is None:
                return True
    return False


def inject_limit(sql: str, limit: int = DEFAULT_LIMIT) -> str:
    """Add `LIMIT <limit>` to a top-level non-aggregate SELECT that has no LIMIT.

    Existing limits and aggregate queries come back unchanged (as the original text).
    """
    tree = sqlglot.parse_one(sql, read="duckdb")
    if isinstance(tree, exp.Select):
        if tree.args.get("limit") is not None or _is_aggregate_query(tree):
            return sql
        return tree.limit(limit).sql(dialect="duckdb")
    if isinstance(tree, exp.SetOperation):
        if tree.args.get("limit") is not None:
            return sql
        return tree.limit(limit).sql(dialect="duckdb")
    return sql
