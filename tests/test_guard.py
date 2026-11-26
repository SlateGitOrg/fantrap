import pytest

from fantrap.guard import violation_codes


def codes(guard, sql, role="analyst"):
    return violation_codes(guard.check(sql, role))


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM invoices",
        "UPDATE invoices SET amount = 0",
        "CREATE TABLE t (a INTEGER)",
        "COPY invoices TO 'out.csv'",
        "ATTACH 'other.db'",
        "INSERT INTO regions VALUES (9, 'x')",
        "PRAGMA database_list",
        "SET memory_limit = '1GB'",
        "INSTALL httpfs",
        "LOAD httpfs",
        "DROP TABLE invoices",
        "MERGE INTO invoices USING regions ON TRUE WHEN MATCHED THEN DELETE",
    ],
)
def test_write_statements(guard, sql):
    assert codes(guard, sql) == ["WRITE_STATEMENT"]


def test_multiple_statements(guard):
    assert codes(guard, "SELECT 1; DROP TABLE invoices") == ["MULTI_STATEMENT"]


def test_trailing_semicolon_is_one_statement(guard):
    assert codes(guard, "SELECT COUNT(*) FROM invoices;") == []


def test_parse_error_has_a_position(guard):
    found = guard.check("SELECT SUM(amount FROM payments", "analyst")
    assert [v.code for v in found] == ["PARSE_ERROR"]
    assert "line 1" in found[0].message


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv('x.csv')",
        "SELECT * FROM read_parquet('x.parquet')",
        "SELECT * FROM read_json_auto('x.json')",
        "SELECT * FROM glob('*')",
        "SELECT getenv('HOME')",
        "SELECT * FROM 'x.csv'",
    ],
)
def test_forbidden_functions(guard, sql):
    assert "FORBIDDEN_FUNCTION" in codes(guard, sql)


def test_row_generating_table_functions_are_allowed(guard):
    assert codes(guard, "SELECT * FROM generate_series(1, 3)") == []


def test_payroll_is_forbidden_for_analyst_and_allowed_for_hr_admin(guard):
    assert codes(guard, "SELECT AVG(salary) FROM payroll") == ["FORBIDDEN_TABLE"]
    assert codes(guard, "SELECT AVG(salary) FROM payroll", "hr_admin") == []


def test_forbidden_table_is_found_inside_a_cte(guard):
    sql = "WITH p AS (SELECT * FROM payroll) SELECT * FROM p"
    assert codes(guard, sql) == ["FORBIDDEN_TABLE"]


def test_hr_admin_cannot_read_invoices(guard):
    assert codes(guard, "SELECT COUNT(*) FROM invoices", "hr_admin") == ["FORBIDDEN_TABLE"]


def test_select_star_expands_into_a_forbidden_column(guard):
    found = guard.check("SELECT * FROM customers", "analyst")
    assert [v.code for v in found] == ["FORBIDDEN_COLUMN"]
    assert "tax_id" in found[0].message


def test_star_is_expanded_through_cte_and_subquery(guard):
    cte = "WITH c AS (SELECT * FROM customers) SELECT name FROM c"
    sub = "SELECT name FROM (SELECT * FROM customers) t"
    alias = "SELECT t.* FROM customers t"
    for sql in (cte, sub, alias):
        assert codes(guard, sql) == ["FORBIDDEN_COLUMN"], sql


def test_star_exclude_and_explicit_columns_pass(guard):
    assert codes(guard, "SELECT * EXCLUDE (tax_id) FROM customers") == []
    assert codes(guard, "SELECT customer_id, name FROM customers") == []
    assert codes(guard, "SELECT COUNT(*) FROM customers") == []


def test_direct_reference_to_a_forbidden_column(guard):
    assert codes(guard, "SELECT c.tax_id FROM customers c") == ["FORBIDDEN_COLUMN"]


def test_unknown_column_suggests_the_closest_name(guard):
    found = guard.check("SELECT total_amount FROM invoices", "analyst")
    assert [v.code for v in found] == ["UNKNOWN_COLUMN"]
    assert found[0].suggestion == "amount"
    assert "amount" in found[0].message


def test_unknown_qualified_column_suggests_too(guard):
    found = guard.check("SELECT i.total_amount FROM invoices i", "analyst")
    assert (found[0].code, found[0].suggestion) == ("UNKNOWN_COLUMN", "amount")


def test_unknown_column_inside_aggregate_with_own_alias(guard):
    found = guard.check("SELECT SUM(total_amount) AS total_amount FROM invoices", "analyst")
    assert [v.code for v in found] == ["UNKNOWN_COLUMN"]


def test_unknown_table_suggests_the_closest_name(guard):
    found = guard.check("SELECT * FROM invoicez", "analyst")
    assert [v.code for v in found] == ["UNKNOWN_TABLE"]
    assert found[0].suggestion == "invoices"


def test_select_aliases_in_order_by_are_not_unknown_columns(guard):
    sql = "SELECT customer_id, SUM(amount) AS t FROM invoices GROUP BY customer_id ORDER BY t"
    assert codes(guard, sql) == []


def test_unknown_role_raises(guard):
    with pytest.raises(ValueError):
        guard.check("SELECT 1", "intern")


def test_fan_trap_message_names_the_edge(guard):
    sql = "SELECT SUM(i.amount) FROM invoices i JOIN shipments s ON s.invoice_id = i.invoice_id"
    found = guard.check(sql, "analyst")
    assert [v.code for v in found] == ["FAN_TRAP"]
    assert "invoices 1:N shipments" in found[0].message


# ---- restricted columns reached without naming them ---------------------------------------

BYPASSES = [
    "SELECT COLUMNS('tax.*') FROM customers",
    "SELECT COLUMNS(*) FROM customers",
    "SELECT c FROM customers c",
    "SELECT customers FROM customers",
    "SELECT to_json(c) FROM customers c",
    "SELECT UNNEST(c) FROM customers c",
    "SELECT (SELECT c) FROM customers c",
    "SELECT #6 FROM customers",
    "SELECT name FROM customers AS t(a, b, c, d, e, name)",
    "SELECT row(c.*) FROM customers c",
]


@pytest.mark.parametrize("sql", BYPASSES)
def test_restricted_column_cannot_be_reached_without_naming_it(guard, sql):
    assert codes(guard, sql) == ["FORBIDDEN_COLUMN"]


def test_the_bypasses_really_leak_tax_ids_without_the_guard(db_path):
    from fantrap.execute import run_query

    leaked = run_query(db_path, "SELECT COLUMNS('tax.*') FROM customers").rows[0][0]
    row = run_query(db_path, "SELECT c FROM customers c").rows[0][0]
    assert leaked == row["tax_id"] and len(leaked) == 10


def test_whole_row_of_an_unrestricted_table_is_fine(guard):
    assert codes(guard, "SELECT i FROM invoices i LIMIT 1") == []
    assert codes(guard, "SELECT r.name FROM regions r") == []


def test_a_derived_table_that_hides_tax_id_may_be_referenced_whole(guard):
    sql = "SELECT t FROM (SELECT customer_id, name FROM customers) t"
    assert codes(guard, sql) == []


def test_role_without_denied_columns_may_use_columns_patterns(guard):
    assert codes(guard, "SELECT COLUMNS('sal.*') FROM payroll", "hr_admin") == []


def test_count_star_and_star_exclude_still_pass(guard):
    assert codes(guard, "SELECT COUNT(*) FROM customers c") == []
    assert codes(guard, "SELECT * EXCLUDE (tax_id) FROM customers c") == []
