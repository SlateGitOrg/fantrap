import duckdb
import pytest

from fantrap.execute import MAX_ROWS, ExecutionError, run_query
from fantrap.rewrite import inject_limit


def test_limit_is_added_to_a_plain_select():
    assert inject_limit("SELECT * FROM invoices").endswith("LIMIT 1000")


def test_existing_limit_is_kept():
    assert inject_limit("SELECT * FROM invoices LIMIT 50") == "SELECT * FROM invoices LIMIT 50"


def test_aggregate_query_is_unchanged():
    sql = "SELECT COUNT(*) FROM invoices"
    assert inject_limit(sql) == sql
    grouped = "SELECT customer_id FROM invoices GROUP BY customer_id"
    assert inject_limit(grouped) == grouped


def test_window_function_is_not_an_aggregate_query():
    out = inject_limit("SELECT invoice_id, SUM(amount) OVER () FROM invoices")
    assert out.endswith("LIMIT 1000")


def test_limit_applies_to_cte_and_union_queries():
    assert inject_limit("WITH x AS (SELECT 1 AS a) SELECT * FROM x").endswith("LIMIT 1000")
    assert inject_limit("SELECT 1 AS a UNION ALL SELECT 2").endswith("LIMIT 1000")


def test_run_query_returns_columns_and_rows(db_path):
    result = run_query(db_path, "SELECT region_id, name FROM regions ORDER BY region_id")
    assert result.columns == ["region_id", "name"]
    assert result.rows[0] == (1, "Northeast")
    assert len(result.rows) == 4 and not result.truncated


def test_row_cap_truncates(db_path):
    result = run_query(db_path, "SELECT * FROM shipments", max_rows=100)
    assert len(result.rows) == 100 and result.truncated
    assert MAX_ROWS == 10_000


def test_default_cap_is_ten_thousand_rows(db_path):
    result = run_query(db_path, "SELECT * FROM range(20000)")
    assert len(result.rows) == 10_000 and result.truncated


def test_delete_sent_straight_to_execute_raises_and_changes_nothing(db_path):
    before = run_query(db_path, "SELECT COUNT(*) FROM invoices").rows[0][0]
    with pytest.raises(ExecutionError, match="read-only"):
        run_query(db_path, "DELETE FROM invoices")
    assert run_query(db_path, "SELECT COUNT(*) FROM invoices").rows[0][0] == before == 3000


def test_file_access_is_disabled_on_the_connection(db_path, tmp_path):
    target = tmp_path / "out.csv"
    with pytest.raises(ExecutionError):
        run_query(db_path, f"COPY invoices TO '{target}'")
    assert not target.exists()
    with pytest.raises(ExecutionError):
        run_query(db_path, "SELECT * FROM read_csv('x.csv')")


def test_engine_errors_are_wrapped(db_path):
    with pytest.raises(ExecutionError, match="does not exist"):
        run_query(db_path, "SELECT * FROM nothing")


def test_missing_database_is_reported(tmp_path):
    with pytest.raises(ExecutionError, match="fantrap seed"):
        run_query(tmp_path / "missing.duckdb", "SELECT 1")


def test_the_warehouse_is_still_writable_by_its_owner(db_path):
    # The read-only guarantee comes from the connection, not from the file.
    con = duckdb.connect(str(db_path))
    try:
        assert con.execute("SELECT COUNT(*) FROM regions").fetchone()[0] == 4
    finally:
        con.close()
