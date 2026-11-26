"""Fan-trap and chasm-trap detection by grain analysis on the sqlglot AST.

For every SELECT scope the analysis builds a join graph from ON/USING and WHERE equalities, orients
each edge with the catalog's key cardinalities, and asks whether an aggregate over a column of
table T is computed on rows that a join has repeated.

Rules:

* An edge is "1:N" when one side of the equality is unique (a primary key, a declared 1:1 child, or
  a derived table grouped by its join key) and the other is not. Two unique sides give "1:1". Two
  non-unique sides give "N:M" (two children of the same parent joined to each other).
* SUM, AVG, COUNT(*) and COUNT(col) without DISTINCT are sensitive to repeated rows. COUNT(DISTINCT
  ...), MIN and MAX are not.
* FAN_TRAP: another source is reached from T by walking the join graph and at least one step goes
  from the "one" side to the "many" side (or is N:M).
* CHASM_TRAP: T is the "one" side of two different 1:N joins.
* EXISTS and IN subqueries are separate scopes, so their correlated equalities never join the
  outer graph and never multiply outer rows.
* An aggregate over an expression that mixes several tables is evaluated at the grain of the finest
  table it references, so only that table is checked (AVG(p.payment_date - i.invoice_date) counts
  payments, not invoices).
* A join that no key equality connects (CROSS JOIN, a range condition, `a.id = b.id + 0`) is
  treated as N:M, unless one side is a single-row aggregate.
* COUNT(*) has no column; it is attributed to the first source in FROM, the entity the query is
  written around.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from .catalog import Catalog

FAN_TRAP = "FAN_TRAP"
CHASM_TRAP = "CHASM_TRAP"

_SENSITIVE = (exp.Sum, exp.Avg, exp.Count)


@dataclass(frozen=True)
class TrapFinding:
    code: str
    message: str
    edge: str
    aggregate: str


@dataclass
class _Source:
    alias: str
    display: str
    table: str | None = None
    scope: Scope | None = None
    catalog: Catalog | None = None
    _unique: object = field(default=None, repr=False)

    def unique(self, columns: frozenset[str]) -> bool:
        if self._unique is None:
            return False
        return bool(self._unique(columns))  # type: ignore[operator]


@dataclass(frozen=True)
class _Edge:
    a: str
    b: str
    kind: str  # "1:1", "1:N" (a is the one side) or "N:M"
    label: str
    on: str

    def multiplies(self, source: str) -> bool:
        """True when walking away from `source` along this edge repeats rows."""
        if self.kind == "N:M":
            return source in (self.a, self.b)
        return self.kind == "1:N" and source == self.a

    def other(self, source: str) -> str:
        return self.b if source == self.a else self.a


# ---- sources -----------------------------------------------------------------------------------


def _from_alias(select: exp.Select) -> str | None:
    from_ = select.args.get("from_") or select.args.get("from")
    return from_.this.alias_or_name if from_ is not None else None


def _derived_source(alias: str, scope: Scope, catalog: Catalog) -> _Source:
    select = scope.expression
    source = _Source(alias=alias, display=alias, scope=scope, catalog=catalog)
    if not isinstance(select, exp.Select):
        return source  # set operations repeat rows freely
    outputs = {e.alias_or_name: e.unalias() for e in select.expressions}
    has_agg = any(select_expr.find(exp.AggFunc) for select_expr in select.expressions)
    group = select.args.get("group")

    if group is not None:
        keys = _group_output_names(group, select)
        source._unique = lambda cols, keys=keys: keys is not None and keys <= cols
        return source
    if has_agg:
        source._unique = lambda cols: True  # one row, whatever the join columns
        return source
    if select.args.get("distinct") is not None:
        names = frozenset(outputs)
        source._unique = lambda cols, names=names: names <= cols
        return source

    inner = _passthrough_source(scope, catalog)
    if inner is not None:
        mapping: dict[str, str] = {}
        for name, expr in outputs.items():
            if isinstance(expr, exp.Column):
                mapping[name] = expr.name
        source.display = inner.display
        source._unique = lambda cols, m=mapping, inner=inner: inner.unique(
            frozenset(m[c] for c in cols if c in m)
        )
    return source


def _group_output_names(group: exp.Group, select: exp.Select) -> frozenset[str] | None:
    """Output column names of the GROUP BY keys, or None when a key is not projected."""
    by_sql = {e.unalias().sql(): e.alias_or_name for e in select.expressions}
    names: set[str] = set()
    for key in group.expressions:
        if isinstance(key, exp.Literal) and key.is_int:
            idx = int(key.name) - 1
            if not 0 <= idx < len(select.expressions):
                return None
            names.add(select.expressions[idx].alias_or_name)
        elif key.sql() in by_sql:
            names.add(by_sql[key.sql()])
        else:
            return None
    return frozenset(names)


def _passthrough_source(scope: Scope, catalog: Catalog) -> _Source | None:
    """The single underlying source of a derived table that only filters and projects."""
    select = scope.expression
    if select.args.get("joins") or select.args.get("group") or select.args.get("having"):
        return None
    selected = scope.selected_sources
    if len(selected) != 1:
        return None
    alias, (_, src) = next(iter(selected.items()))
    return _make_source(alias, src, catalog)


def _make_source(alias: str, src: exp.Table | Scope, catalog: Catalog) -> _Source:
    if isinstance(src, Scope):
        return _derived_source(alias, src, catalog)
    table = src.name.lower()
    return _Source(
        alias=alias,
        display=table,
        table=table,
        catalog=catalog,
        _unique=lambda cols, t=table: catalog.is_unique(t, set(cols)),
    )


# ---- join graph --------------------------------------------------------------------------------


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    if isinstance(node, exp.And):
        return _conjuncts(node.left) + _conjuncts(node.right)
    if isinstance(node, exp.Paren):
        return _conjuncts(node.this)
    return [node]


def _equalities(select: exp.Select) -> list[tuple[exp.Column, exp.Column]]:
    conditions: list[exp.Expression] = []
    for join in select.args.get("joins") or []:
        on = join.args.get("on")
        if on is not None:
            conditions.extend(_conjuncts(on))
    where = select.args.get("where")
    if where is not None:
        conditions.extend(_conjuncts(where.this))
    pairs = []
    for cond in conditions:
        if isinstance(cond, exp.EQ):
            left, right = cond.left, cond.right
            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                if left.table and right.table:
                    pairs.append((left, right))
    return pairs


def _build_edges(select: exp.Select, sources: dict[str, _Source]) -> list[_Edge]:
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for left, right in _equalities(select):
        a, b = left.table, right.table
        if a == b or a not in sources or b not in sources:
            continue
        if a <= b:
            grouped.setdefault((a, b), []).append((left.name, right.name))
        else:
            grouped.setdefault((b, a), []).append((right.name, left.name))
    edges = []
    for (a, b), cols in grouped.items():
        cols_a = frozenset(c[0] for c in cols)
        cols_b = frozenset(c[1] for c in cols)
        unique_a = sources[a].unique(cols_a)
        unique_b = sources[b].unique(cols_b)
        on = " AND ".join(f"{a}.{ca} = {b}.{cb}" for ca, cb in cols)
        if unique_a and unique_b:
            edges.append(_Edge(a, b, "1:1", f"{sources[a].display} 1:1 {sources[b].display}", on))
        elif unique_a:
            edges.append(_Edge(a, b, "1:N", f"{sources[a].display} 1:N {sources[b].display}", on))
        elif unique_b:
            edges.append(_Edge(b, a, "1:N", f"{sources[b].display} 1:N {sources[a].display}", on))
        else:
            edges.append(_Edge(a, b, "N:M", f"{sources[a].display} N:M {sources[b].display}", on))
    return edges + _unkeyed_edges(sources, edges)


def _unkeyed_edges(sources: dict[str, _Source], edges: list[_Edge]) -> list[_Edge]:
    """N:M edges for sources that no key equality joins to the rest (CROSS JOIN, range joins,
    equalities on expressions such as `a.id = b.id + 0`). Every row of one side pairs with an
    unknown number of rows of the other, so the result is treated as repeating rows. A source
    that is known to return a single row (an aggregate without GROUP BY) is harmless."""
    parent = {alias: alias for alias in sources}

    def find(alias: str) -> str:
        while parent[alias] != alias:
            parent[alias] = parent[parent[alias]]
            alias = parent[alias]
        return alias

    for edge in edges:
        parent[find(edge.a)] = find(edge.b)
    components: dict[str, list[str]] = {}
    for alias in sources:
        components.setdefault(find(alias), []).append(alias)
    multi = [
        g for g in components.values() if not (len(g) == 1 and sources[g[0]].unique(frozenset()))
    ]
    extra: list[_Edge] = []
    for group in multi[1:]:
        a, b = multi[0][0], group[0]
        label = f"{sources[a].display} N:M {sources[b].display}"
        extra.append(_Edge(a, b, "N:M", label, "no key equality joins these sources"))
    return extra


def _descends(start: str, target: str, edges: list[_Edge]) -> bool:
    """True when `target` is reached from `start` using only one-to-many or one-to-one steps."""
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for edge in edges:
            if node not in (edge.a, edge.b):
                continue
            nxt = edge.other(node)
            downhill = edge.kind == "1:1" or (edge.kind == "1:N" and edge.a == node)
            if downhill and nxt not in seen:
                if nxt == target:
                    return True
                seen.add(nxt)
                queue.append(nxt)
    return False


def _multiplying_edges(start: str, edges: list[_Edge]) -> list[_Edge]:
    """Edges that repeat T's rows, found by walking the join graph outward from `start`."""
    found: list[_Edge] = []
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for edge in edges:
            if node not in (edge.a, edge.b):
                continue
            nxt = edge.other(node)
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
            if edge.multiplies(node):
                found.append(edge)
    return found


# ---- aggregates --------------------------------------------------------------------------------


def _sensitive(agg: exp.Expression) -> bool:
    if not isinstance(agg, _SENSITIVE):
        return False
    return not (isinstance(agg, exp.Count) and isinstance(agg.this, exp.Distinct))


def _referenced(agg: exp.Expression, sources: dict[str, _Source], select: exp.Select) -> list[str]:
    aliases = []
    for col in agg.find_all(exp.Column):
        if col.table in sources and col.table not in aliases:
            aliases.append(col.table)
    if aliases:
        return aliases
    first = _from_alias(select)  # COUNT(*), COUNT(1): the query's driving entity
    return [first] if first in sources else []


def _scan_scope(scope: Scope, catalog: Catalog) -> list[TrapFinding]:
    select = scope.expression
    sources = {
        alias: _make_source(alias, src, catalog)
        for alias, (_, src) in scope.selected_sources.items()
    }
    if len(sources) < 2:
        return []
    edges = _build_edges(select, sources)
    if not edges:
        return []  # sources are all single-row or there are none to join

    findings: list[TrapFinding] = []
    seen: set[tuple[str, str, str]] = set()
    for agg in select.find_all(*_SENSITIVE):
        if agg.parent_select is not select or not _sensitive(agg):
            continue
        refs = _referenced(agg, sources, select)
        agg_sql = agg.sql("duckdb")
        for table in refs:
            if any(other != table and _descends(table, other, edges) for other in refs):
                continue  # the expression is evaluated at a finer grain than this table
            children = [e for e in edges if e.kind == "1:N" and e.a == table]
            if len({e.b for e in children}) >= 2:
                labels = " and ".join(e.label for e in children)
                key = (CHASM_TRAP, labels, agg_sql)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    TrapFinding(
                        CHASM_TRAP,
                        f"{agg_sql} is computed over {sources[table].display}, which is the "
                        f"'one' side of two 1:N joins ({labels}). The two child tables multiply "
                        f"each other's rows, so the result is inflated. Aggregate each child in "
                        f"a subquery grouped by the join key before joining, or use "
                        f"COUNT(DISTINCT key).",
                        labels,
                        agg_sql,
                    )
                )
                continue
            fans = _multiplying_edges(table, edges)
            if fans:
                labels = "; ".join(dict.fromkeys(e.label for e in fans))
                key = (FAN_TRAP, labels, agg_sql)
                if key in seen:
                    continue
                seen.add(key)
                on = fans[0].on
                findings.append(
                    TrapFinding(
                        FAN_TRAP,
                        f"{agg_sql} is computed over {sources[table].display} but the join "
                        f"{labels} ({on}) repeats each of its rows once per matching row on the "
                        f"other side, so the result is inflated. Aggregate the many side in a "
                        f"subquery grouped by the join key, filter with EXISTS, or use "
                        f"COUNT(DISTINCT key).",
                        labels,
                        agg_sql,
                    )
                )
    return findings


def find_traps(ast: exp.Expression, catalog: Catalog) -> list[TrapFinding]:
    """Trap findings for a qualified query AST, across every SELECT scope."""
    findings: list[TrapFinding] = []
    for scope in traverse_scope(ast):
        if isinstance(scope.expression, exp.Select):
            findings.extend(_scan_scope(scope, catalog))
    return findings
