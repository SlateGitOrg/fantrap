"""Semantic catalog: tables, typed columns, keys, cardinalities, measures and role allow-lists."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CATALOG_PATH = Path("catalog.yaml")


class TableSpec(BaseModel):
    description: str = ""
    primary_key: list[str]
    columns: dict[str, str]


class Relationship(BaseModel):
    """A foreign key. `parent` is the referenced key, `child` the referencing column."""

    parent: str
    child: str
    cardinality: Literal["1:N", "1:1"] = "1:N"

    @property
    def parent_table(self) -> str:
        return self.parent.split(".")[0]

    @property
    def parent_column(self) -> str:
        return self.parent.split(".")[1]

    @property
    def child_table(self) -> str:
        return self.child.split(".")[0]

    @property
    def child_column(self) -> str:
        return self.child.split(".")[1]

    def label(self) -> str:
        return f"{self.parent_table} {self.cardinality} {self.child_table}"


class RoleSpec(BaseModel):
    description: str = ""
    allow_tables: list[str]
    deny_columns: list[str] = Field(default_factory=list)


class Catalog(BaseModel):
    tables: dict[str, TableSpec]
    relationships: list[Relationship] = Field(default_factory=list)
    measures: dict[str, list[str]] = Field(default_factory=dict)
    roles: dict[str, RoleSpec]

    @model_validator(mode="after")
    def _check_references(self) -> Catalog:
        for name, table in self.tables.items():
            for key in table.primary_key:
                if key not in table.columns:
                    raise ValueError(f"{name}: primary key column {key} is not a column")
        for rel in self.relationships:
            for ref in (rel.parent, rel.child):
                self._require_column(ref)
            if rel.parent_column not in self.tables[rel.parent_table].primary_key:
                raise ValueError(f"{rel.parent} is the parent of a relationship but not a key")
        for table, cols in self.measures.items():
            for col in cols:
                self._require_column(f"{table}.{col}")
        for role, spec in self.roles.items():
            for table in spec.allow_tables:
                if table not in self.tables:
                    raise ValueError(f"role {role}: unknown table {table}")
            for ref in spec.deny_columns:
                self._require_column(ref)
        return self

    def _require_column(self, ref: str) -> None:
        table, _, column = ref.partition(".")
        if table not in self.tables or column not in self.tables[table].columns:
            raise ValueError(f"unknown column reference {ref}")

    # ---- lookups -------------------------------------------------------------------------

    def table_names(self) -> list[str]:
        return list(self.tables)

    def columns(self, table: str) -> list[str]:
        return list(self.tables[table].columns)

    def schema(self) -> dict[str, dict[str, str]]:
        """Schema mapping in the shape sqlglot's optimizer expects."""
        return {name: dict(spec.columns) for name, spec in self.tables.items()}

    def is_unique(self, table: str, columns: set[str] | frozenset[str]) -> bool:
        """True when `columns` pin down at most one row of `table`.

        A set is unique when it contains the primary key, or when it contains the child column of
        a declared 1:1 relationship.
        """
        spec = self.tables.get(table)
        if spec is None:
            return False
        if spec.primary_key and set(spec.primary_key) <= set(columns):
            return True
        return any(
            rel.cardinality == "1:1" and rel.child_table == table and rel.child_column in columns
            for rel in self.relationships
        )

    def relationship(
        self, table_a: str, col_a: str, table_b: str, col_b: str
    ) -> Relationship | None:
        """The declared relationship joining the two columns, in either orientation."""
        for rel in self.relationships:
            forward = (rel.parent_table, rel.parent_column, rel.child_table, rel.child_column)
            if forward == (table_a, col_a, table_b, col_b):
                return rel
            if forward == (table_b, col_b, table_a, col_a):
                return rel
        return None

    def measure_columns(self, table: str) -> list[str]:
        return self.measures.get(table, [])

    # ---- roles ---------------------------------------------------------------------------

    def role(self, name: str) -> RoleSpec:
        if name not in self.roles:
            known = ", ".join(sorted(self.roles))
            raise ValueError(f"unknown role '{name}' (known roles: {known})")
        return self.roles[name]

    def allowed_tables(self, role: str) -> list[str]:
        spec = self.role(role)
        return [t for t in self.tables if t in spec.allow_tables]

    def denied_columns(self, role: str) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for ref in self.role(role).deny_columns:
            table, _, column = ref.partition(".")
            out.add((table, column))
        return out

    def schema_prompt(self, role: str) -> str:
        """Schema description handed to the model: only what `role` may read."""
        denied = self.denied_columns(role)
        lines = ["DuckDB tables you may query:"]
        for table in self.allowed_tables(role):
            spec = self.tables[table]
            cols = [
                f"{col} {typ}" for col, typ in spec.columns.items() if (table, col) not in denied
            ]
            lines.append(
                f"- {table}({', '.join(cols)})  primary key: {', '.join(spec.primary_key)}"
            )
            if spec.description:
                lines.append(f"    {spec.description}")
        allowed = set(self.allowed_tables(role))
        rels = [
            rel
            for rel in self.relationships
            if rel.parent_table in allowed and rel.child_table in allowed
        ]
        if rels:
            lines.append("Joins (parent key -> child column, cardinality):")
            for rel in rels:
                lines.append(f"- {rel.parent} -> {rel.child}  ({rel.label()})")
        measures = [
            f"{t}.{c}"
            for t in self.allowed_tables(role)
            for c in self.measure_columns(t)
            if (t, c) not in denied
        ]
        if measures:
            lines.append("Measure columns: " + ", ".join(measures))
            lines.append(
                "A measure on the 'one' side of a 1:N join repeats once per child row. Aggregate "
                "the child table in a subquery grouped by the join key before joining."
            )
        return "\n".join(lines)


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> Catalog:
    with open(path, encoding="utf-8") as handle:
        return Catalog.model_validate(yaml.safe_load(handle))
