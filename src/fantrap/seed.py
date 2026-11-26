"""Seeded synthetic accounts-receivable warehouse for a B2B wholesale distributor.

All data is synthetic and deterministic for a given seed. The generator plants the shapes the
guard has to cope with: exactly 35% of invoices ship in 2-3 partial shipments, payments arrive in
installments, 8% of invoices are unpaid and past due, and 5% are only partly paid.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np

AS_OF = date(2025, 12, 31)
START = date(2024, 1, 1)
N_CUSTOMERS = 200
N_INVOICES = 3000
N_CREDIT_NOTES = 150
N_PAYROLL = 40
SPLIT_SHARE = 0.35
UNPAID_SHARE = 0.08
PARTIAL_SHARE = 0.05

REGIONS = ["Northeast", "Southeast", "Midwest", "West"]
SEGMENTS = ["Industrial", "Retail", "Foodservice", "Healthcare"]
DEPARTMENTS = ["Finance", "Operations", "Sales", "Warehouse", "IT"]
_ADJECTIVES = ["Harbor", "Summit", "Prairie", "Cedar", "Granite", "Lakeside", "Redwood", "Meridian"]
_NOUNS = ["Supply", "Provisions", "Trading", "Distribution", "Goods", "Wholesale", "Freight"]

DDL = {
    "regions": "region_id INTEGER PRIMARY KEY, name VARCHAR",
    "customers": (
        "customer_id INTEGER PRIMARY KEY, name VARCHAR, region_id INTEGER, segment VARCHAR, "
        "terms_days INTEGER, tax_id VARCHAR"
    ),
    "invoices": (
        "invoice_id INTEGER PRIMARY KEY, customer_id INTEGER, invoice_date DATE, due_date DATE, "
        "amount DOUBLE"
    ),
    "shipments": (
        "shipment_id INTEGER PRIMARY KEY, invoice_id INTEGER, ship_date DATE, units INTEGER, "
        "freight_cost DOUBLE"
    ),
    "payments": (
        "payment_id INTEGER PRIMARY KEY, invoice_id INTEGER, payment_date DATE, amount DOUBLE"
    ),
    "credit_notes": (
        "credit_note_id INTEGER PRIMARY KEY, invoice_id INTEGER, issued_date DATE, amount DOUBLE"
    ),
    "payroll": (
        "employee_id INTEGER PRIMARY KEY, department VARCHAR, region_id INTEGER, salary DOUBLE"
    ),
}


def _split_cents(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    """Split `total` cents into `parts` positive pieces that add up exactly."""
    if parts == 1:
        return [total]
    weights = rng.dirichlet(np.ones(parts))
    pieces = [max(1, int(total * w)) for w in weights[:-1]]
    pieces.append(total - sum(pieces))
    return pieces


def _build_customers(rng: np.random.Generator) -> list[tuple]:
    rows = []
    segment_p = [0.35, 0.30, 0.20, 0.15]
    for cid in range(1, N_CUSTOMERS + 1):
        name = (
            f"{_ADJECTIVES[int(rng.integers(len(_ADJECTIVES)))]} "
            f"{_NOUNS[int(rng.integers(len(_NOUNS)))]} {cid:03d}"
        )
        rows.append(
            (
                cid,
                name,
                int(rng.integers(1, len(REGIONS) + 1)),
                str(rng.choice(SEGMENTS, p=segment_p)),
                int(rng.choice([30, 45, 60])),
                f"{int(rng.integers(10, 99))}-{int(rng.integers(1_000_000, 9_999_999))}",
            )
        )
    return rows


def _build_invoices(rng: np.random.Generator, customers: list[tuple]) -> list[tuple]:
    weights = rng.lognormal(0.0, 0.8, N_CUSTOMERS)
    weights = weights / weights.sum()
    customer_ids = rng.choice(np.arange(1, N_CUSTOMERS + 1), size=N_INVOICES, p=weights)
    offsets = np.sort(rng.integers(0, (AS_OF - START).days + 1, N_INVOICES))
    amounts = np.maximum(np.round(rng.lognormal(8.0, 0.9, N_INVOICES) * 100), 5000).astype(np.int64)
    terms = {c[0]: c[4] for c in customers}
    rows = []
    for i in range(N_INVOICES):
        cid = int(customer_ids[i])
        inv_date = START + timedelta(days=int(offsets[i]))
        rows.append(
            (i + 1, cid, inv_date, inv_date + timedelta(days=terms[cid]), int(amounts[i]) / 100)
        )
    return rows


def _build_shipments(rng: np.random.Generator, invoices: list[tuple]) -> list[tuple]:
    n_split = round(SPLIT_SHARE * len(invoices))
    split = set(rng.choice(len(invoices), n_split, replace=False).tolist())
    rows = []
    shipment_id = 1
    for idx, (invoice_id, _, inv_date, _, amount) in enumerate(invoices):
        parts = 1
        if idx in split:
            parts = 2 if rng.random() < 0.6 else 3
        total_units = max(3, int(round(amount / float(rng.uniform(15, 60)))))
        base, extra = divmod(total_units, parts)
        ship_date = inv_date + timedelta(days=int(rng.integers(0, 11)))
        for part in range(parts):
            units = base + (extra if part == parts - 1 else 0)
            freight = round(units * float(rng.uniform(0.4, 1.2)), 2)
            rows.append((shipment_id, invoice_id, min(ship_date, AS_OF), units, freight))
            shipment_id += 1
            ship_date += timedelta(days=int(rng.integers(1, 15)))
    return rows


def _build_payments(rng: np.random.Generator, invoices: list[tuple]) -> list[tuple]:
    n = len(invoices)
    eligible = [i for i, inv in enumerate(invoices) if inv[3] <= date(2025, 11, 30)]
    unpaid = set(rng.choice(eligible, round(UNPAID_SHARE * n), replace=False).tolist())
    rest = [i for i in range(n) if i not in unpaid]
    partial = set(rng.choice(rest, round(PARTIAL_SHARE * n), replace=False).tolist())
    rows = []
    payment_id = 1
    for idx, (invoice_id, _, inv_date, due_date, amount) in enumerate(invoices):
        if idx in unpaid:
            continue
        total_cents = round(amount * 100)
        if idx in partial:
            total_cents = int(total_cents * float(rng.uniform(0.4, 0.9)))
            parts = int(rng.choice([1, 2], p=[0.7, 0.3]))
        else:
            parts = int(rng.choice([1, 2, 3], p=[0.67, 0.28, 0.05]))
        pay_date = due_date + timedelta(days=int(round(float(rng.normal(0, 12)))))
        for cents in _split_cents(rng, total_cents, parts):
            pay_date = max(pay_date, inv_date + timedelta(days=1))
            rows.append((payment_id, invoice_id, min(pay_date, AS_OF), cents / 100))
            payment_id += 1
            pay_date += timedelta(days=int(rng.integers(5, 30)))
    return rows


def _build_credit_notes(rng: np.random.Generator, invoices: list[tuple]) -> list[tuple]:
    chosen = sorted(rng.choice(len(invoices), N_CREDIT_NOTES, replace=False).tolist())
    rows = []
    for note_id, idx in enumerate(chosen, start=1):
        invoice_id, _, inv_date, _, amount = invoices[idx]
        issued = min(inv_date + timedelta(days=int(rng.integers(5, 60))), AS_OF)
        rows.append(
            (note_id, invoice_id, issued, round(amount * float(rng.uniform(0.05, 0.30)), 2))
        )
    return rows


def _build_payroll(rng: np.random.Generator) -> list[tuple]:
    rows = []
    for emp in range(1, N_PAYROLL + 1):
        salary = max(40000.0, round(float(rng.normal(72000, 16000)), -2))
        rows.append(
            (
                emp,
                str(rng.choice(DEPARTMENTS)),
                int(rng.integers(1, len(REGIONS) + 1)),
                salary,
            )
        )
    return rows


def _insert(
    con: duckdb.DuckDBPyConnection, table: str, rows: list[tuple], chunk: int = 400
) -> None:
    if not rows:
        return
    placeholder = "(" + ", ".join("?" * len(rows[0])) + ")"
    for start in range(0, len(rows), chunk):
        batch = rows[start : start + chunk]
        params = [value for row in batch for value in row]
        con.execute(f"INSERT INTO {table} VALUES " + ", ".join([placeholder] * len(batch)), params)


def build_warehouse(seed: int, out: str | Path) -> dict[str, int]:
    """Write the warehouse to `out` (replacing any file there) and return the row counts."""
    rng = np.random.default_rng(seed)
    customers = _build_customers(rng)
    invoices = _build_invoices(rng, customers)
    tables: dict[str, list[tuple]] = {
        "regions": [(i + 1, name) for i, name in enumerate(REGIONS)],
        "customers": customers,
        "invoices": invoices,
        "shipments": _build_shipments(rng, invoices),
        "payments": _build_payments(rng, invoices),
        "credit_notes": _build_credit_notes(rng, invoices),
        "payroll": _build_payroll(rng),
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    try:
        for table, ddl in DDL.items():
            con.execute(f"CREATE TABLE {table} ({ddl})")
            _insert(con, table, tables[table])
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return {table: len(rows) for table, rows in tables.items()}
