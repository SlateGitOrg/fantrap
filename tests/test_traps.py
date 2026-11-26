from collections import Counter
from pathlib import Path

import pytest
import yaml

CASES = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "trap_cases.yaml").read_text(encoding="utf-8")
)["cases"]


def classify(guard, sql):
    found = guard.check(sql, "analyst")
    codes = {v.code for v in found}
    assert codes <= {"FAN_TRAP", "CHASM_TRAP"}, f"unexpected violations: {found}"
    if "CHASM_TRAP" in codes:
        return "chasm"
    return "fan" if "FAN_TRAP" in codes else "safe"


def test_fixture_has_the_documented_mix():
    assert Counter(c["label"] for c in CASES) == {"fan": 12, "chasm": 4, "safe": 14}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_each_case_is_classified_correctly(guard, case):
    assert classify(guard, case["sql"]) == case["label"]


def test_detector_scores_30_of_30(guard):
    hits = sum(classify(guard, c["sql"]) == c["label"] for c in CASES)
    assert (hits, len(CASES)) == (30, 30)


def test_chasm_message_names_both_edges(guard):
    sql = CASES[[c["id"] for c in CASES].index("chasm_sum_two_children")]["sql"]
    message = guard.check(sql, "analyst")[0].message
    assert "invoices 1:N shipments" in message and "invoices 1:N payments" in message


def test_fan_trap_through_a_cte_is_found(guard):
    sql = (
        "WITH inv AS (SELECT invoice_id, amount FROM invoices) "
        "SELECT SUM(inv.amount) FROM inv JOIN shipments s ON s.invoice_id = inv.invoice_id"
    )
    assert classify(guard, sql) == "fan"


def test_derived_table_grouped_by_more_than_the_join_key_is_not_one_to_one(guard):
    sql = (
        "SELECT SUM(i.amount) FROM invoices i JOIN (SELECT invoice_id, payment_date, "
        "SUM(amount) AS paid FROM payments GROUP BY invoice_id, payment_date) p "
        "ON p.invoice_id = i.invoice_id"
    )
    assert classify(guard, sql) == "fan"


def test_trap_inside_a_subquery_is_found(guard):
    sql = (
        "SELECT * FROM (SELECT SUM(i.amount) AS t FROM invoices i "
        "JOIN shipments s ON s.invoice_id = i.invoice_id) x"
    )
    assert classify(guard, sql) == "fan"


def test_correlated_exists_does_not_join_the_outer_scope(guard):
    sql = (
        "SELECT COUNT(*) FROM invoices i JOIN customers c ON c.customer_id = i.customer_id "
        "WHERE EXISTS (SELECT 1 FROM payments p WHERE p.invoice_id = i.invoice_id)"
    )
    assert classify(guard, sql) == "safe"


# ---- joins that no key equality connects ----------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(i.amount) FROM invoices i CROSS JOIN shipments s",
        "SELECT SUM(i.amount) FROM invoices i, shipments s",
        "SELECT SUM(i.amount) FROM invoices i JOIN shipments s ON i.invoice_id = s.invoice_id + 0",
        "SELECT SUM(i.amount) FROM invoices i JOIN shipments s ON s.ship_date >= i.invoice_date",
        "SELECT SUM(i.amount) FROM invoices i JOIN shipments s ON TRUE",
        "SELECT COUNT(*) FROM invoices i CROSS JOIN customers c",
        "SELECT SUM(c.terms_days) FROM customers c CROSS JOIN regions r",
    ],
)
def test_unkeyed_joins_are_flagged(guard, sql):
    found = guard.check(sql, "analyst")
    assert [v.code for v in found] == ["FAN_TRAP"], sql
    assert "N:M" in found[0].message


def test_unkeyed_join_flags_the_aggregate_on_either_side(guard):
    sql = "SELECT SUM(s.freight_cost) FROM invoices i CROSS JOIN shipments s"
    assert classify(guard, sql) == "fan"


def test_unkeyed_join_is_found_next_to_keyed_joins(guard):
    sql = (
        "SELECT SUM(i.amount) FROM invoices i JOIN customers c ON c.customer_id = i.customer_id "
        "CROSS JOIN regions r"
    )
    assert classify(guard, sql) == "fan"


def test_unkeyed_join_executed_really_inflates_the_total(db_path):
    from fantrap.execute import run_query

    plain = run_query(db_path, "SELECT SUM(amount) FROM invoices").rows[0][0]
    crossed = run_query(
        db_path, "SELECT SUM(i.amount) FROM invoices i CROSS JOIN shipments s"
    ).rows[0][0]
    assert crossed > plain * 1000


def test_cross_join_with_a_single_row_aggregate_is_safe(guard):
    sql = (
        "SELECT SUM(i.amount / t.total) FROM invoices i "
        "CROSS JOIN (SELECT SUM(amount) AS total FROM invoices) t"
    )
    assert classify(guard, sql) == "safe"


def test_keyed_joins_stay_safe_when_every_source_is_connected(guard):
    sql = (
        "SELECT SUM(i.amount) FROM invoices i JOIN customers c ON c.customer_id = i.customer_id "
        "JOIN regions r ON r.region_id = c.region_id"
    )
    assert classify(guard, sql) == "safe"
