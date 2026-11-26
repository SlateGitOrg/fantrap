import duckdb

from fantrap.seed import AS_OF, build_warehouse


def _scalar(path, sql):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(sql).fetchone()[0]
    finally:
        con.close()


def test_same_seed_gives_identical_counts_and_total(tmp_path):
    first = build_warehouse(7, tmp_path / "a.duckdb")
    second = build_warehouse(7, tmp_path / "b.duckdb")
    assert first == second
    total_a = _scalar(tmp_path / "a.duckdb", "SELECT SUM(amount) FROM invoices")
    total_b = _scalar(tmp_path / "b.duckdb", "SELECT SUM(amount) FROM invoices")
    assert total_a == total_b


def test_different_seed_changes_the_data(tmp_path, db_path):
    build_warehouse(8, tmp_path / "c.duckdb")
    other = _scalar(tmp_path / "c.duckdb", "SELECT SUM(amount) FROM invoices")
    assert other != _scalar(db_path, "SELECT SUM(amount) FROM invoices")


def test_fixed_table_sizes(db_path):
    for table, n in [
        ("regions", 4),
        ("customers", 200),
        ("invoices", 3000),
        ("credit_notes", 150),
        ("payroll", 40),
    ]:
        assert _scalar(db_path, f"SELECT COUNT(*) FROM {table}") == n


def test_exactly_35_percent_of_invoices_have_two_or_more_shipments(db_path):
    split = _scalar(
        db_path,
        "SELECT COUNT(*) FROM (SELECT invoice_id FROM shipments GROUP BY invoice_id "
        "HAVING COUNT(*) >= 2)",
    )
    assert split / 3000 == 0.35


def test_split_invoices_have_two_or_three_shipments_and_all_invoices_ship(db_path):
    counts = _scalar(
        db_path,
        "SELECT MIN(n) || '-' || MAX(n) "
        "FROM (SELECT COUNT(*) n FROM shipments GROUP BY invoice_id)",
    )
    assert counts == "1-3"
    assert _scalar(db_path, "SELECT COUNT(DISTINCT invoice_id) FROM shipments") == 3000


def test_shipment_and_payment_volumes_are_close_to_the_spec(db_path):
    assert 4300 <= _scalar(db_path, "SELECT COUNT(*) FROM shipments") <= 4700
    assert 3600 <= _scalar(db_path, "SELECT COUNT(*) FROM payments") <= 4000


def test_eight_percent_of_invoices_are_unpaid_and_past_due(db_path):
    unpaid = _scalar(
        db_path,
        "SELECT COUNT(*) FROM invoices i WHERE NOT EXISTS "
        "(SELECT 1 FROM payments p WHERE p.invoice_id = i.invoice_id)",
    )
    assert unpaid == 240
    past_due = _scalar(
        db_path,
        "SELECT COUNT(*) FROM invoices i WHERE due_date < DATE '2025-12-31' AND NOT EXISTS "
        "(SELECT 1 FROM payments p WHERE p.invoice_id = i.invoice_id)",
    )
    assert past_due == 240


def test_payments_can_be_partial_or_installments(db_path):
    partial = _scalar(
        db_path,
        "SELECT COUNT(*) FROM (SELECT i.invoice_id FROM invoices i JOIN payments p "
        "ON p.invoice_id = i.invoice_id GROUP BY i.invoice_id, i.amount "
        "HAVING SUM(p.amount) < i.amount - 0.01)",
    )
    assert partial == 150
    installments = _scalar(
        db_path,
        "SELECT COUNT(*) FROM (SELECT invoice_id FROM payments GROUP BY invoice_id "
        "HAVING COUNT(*) >= 2)",
    )
    assert installments > 500


def test_dates_cover_2024_and_2025_and_amounts_are_skewed(db_path):
    assert str(_scalar(db_path, "SELECT MIN(invoice_date) FROM invoices")) >= "2024-01-01"
    assert str(_scalar(db_path, "SELECT MAX(invoice_date) FROM invoices")) <= str(AS_OF)
    mean = _scalar(db_path, "SELECT AVG(amount) FROM invoices")
    median = _scalar(db_path, "SELECT MEDIAN(amount) FROM invoices")
    assert mean > median * 1.2  # lognormal: the mean sits above the median


def test_fan_out_is_real_in_the_data(db_path):
    plain = _scalar(db_path, "SELECT SUM(amount) FROM invoices")
    joined = _scalar(
        db_path,
        "SELECT SUM(i.amount) FROM invoices i JOIN shipments s ON s.invoice_id = i.invoice_id",
    )
    assert joined > plain * 1.2
