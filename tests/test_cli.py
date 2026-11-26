from typer.testing import CliRunner

from fantrap.cli import app

runner = CliRunner()


def test_seed_command_writes_the_warehouse(tmp_path):
    out = tmp_path / "w.duckdb"
    result = runner.invoke(app, ["seed", "--seed", "7", "--out", str(out)])
    assert result.exit_code == 0 and out.exists()
    assert "invoices" in result.output and "3,000" in result.output


def test_check_prints_violations_and_exits_nonzero(tmp_path):
    sql = "SELECT SUM(i.amount) FROM invoices i JOIN shipments s ON s.invoice_id = i.invoice_id"
    result = runner.invoke(app, ["check", sql, "--role", "analyst"])
    assert result.exit_code == 1
    assert "FAN_TRAP" in result.output and "invoices 1:N shipments" in result.output


def test_check_passes_clean_sql():
    result = runner.invoke(app, ["check", "SELECT COUNT(*) FROM invoices"])
    assert result.exit_code == 0 and "OK" in result.output


def test_check_compares_with_gold(db_path):
    sql = "SELECT SUM(i.amount) FROM invoices i JOIN shipments s ON s.invoice_id = i.invoice_id"
    gold = "SELECT SUM(amount) FROM invoices"
    result = runner.invoke(app, ["check", sql, "--gold", gold, "--db", str(db_path)])
    assert "gold total" in result.output and "query returns" in result.output


def test_check_with_hr_admin_role():
    result = runner.invoke(app, ["check", "SELECT AVG(salary) FROM payroll", "--role", "hr_admin"])
    assert result.exit_code == 0


def test_ask_runs_the_scripted_pipeline(db_path):
    result = runner.invoke(
        app, ["ask", "How many customers are in each segment?", "--db", str(db_path)]
    )
    assert result.exit_code == 0
    assert "Industrial" in result.output and "PARSE_ERROR" in result.output


def test_ask_refusal_exits_nonzero(db_path):
    result = runner.invoke(
        app, ["ask", "What is the average salary across the company?", "--db", str(db_path)]
    )
    assert result.exit_code == 1 and "REFUSED: FORBIDDEN_TABLE" in result.output


def test_eval_command_writes_reports(db_path, tmp_path):
    result = runner.invoke(app, ["eval", "--db", str(db_path), "--out", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "eval.md").exists() and (tmp_path / "eval.json").exists()
    assert "32/35" in result.output
