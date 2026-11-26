import json

import duckdb
import pytest

from fantrap.evaluate import (
    exact_match,
    has_order_by,
    load_questions,
    normalize_sql,
    results_match,
    run_eval,
    write_reports,
)
from fantrap.execute import QueryResult, run_query
from fantrap.model import ScriptedModel


def result(rows, columns=("a", "b")):
    return QueryResult(list(columns), rows)


# ---- comparator ----------------------------------------------------------------------------


def test_reordered_rows_and_renamed_columns_are_equal():
    got = result([(2, "y"), (1, "x")], ("p", "q"))
    want = result([(1, "x"), (2, "y")], ("a", "b"))
    assert results_match(got, want)


def test_different_order_is_unequal_when_gold_has_order_by():
    got = result([(2, "y"), (1, "x")])
    want = result([(1, "x"), (2, "y")])
    assert not results_match(got, want, ordered=True)
    assert results_match(want, want, ordered=True)


def test_floats_within_relative_tolerance_are_equal():
    assert results_match(result([(1_000_000.0, "x")]), result([(1_000_000.5, "x")]))
    assert not results_match(result([(1_000_000.0, "x")]), result([(1_000_002.0, "x")]))
    assert results_match(result([(0.0, "x")]), result([(1e-12, "x")]))


def test_ints_and_floats_compare_as_numbers():
    assert results_match(result([(3, "x")]), result([(3.0, "x")]))


def test_multiset_semantics_count_duplicates():
    assert not results_match(result([(1, "x"), (1, "x")]), result([(1, "x"), (2, "y")]))
    assert not results_match(result([(1, "x")]), result([(1, "x"), (1, "x")]))


def test_column_order_may_differ():
    got = result([("x", 1.0), ("y", 2.0)])
    want = result([(1.0, "x"), (2.0, "y")])
    assert results_match(got, want)
    assert not results_match(result([("x", 1.0), ("y", 3.0)]), want)


def test_different_width_or_values_are_unequal():
    assert not results_match(result([(1, 2)]), result([(1, 2, 3)], ("a", "b", "c")))
    assert not results_match(result([(1, "x")]), result([(1, "z")]))


def test_empty_results_match_only_on_width():
    assert results_match(result([]), result([]))


def test_gold_order_by_detection():
    assert has_order_by("SELECT a FROM t ORDER BY a")
    assert not has_order_by("SELECT a FROM t")


# ---- exact match ---------------------------------------------------------------------------


def test_exact_match_ignores_case_whitespace_and_semicolon():
    assert exact_match("select a  from T where a=1;", "SELECT a FROM t WHERE a = 1")
    assert normalize_sql("SELECT 1") == normalize_sql("select   1;")


def test_exact_match_rejects_a_rewrite_with_the_same_answer():
    cte = "WITH x AS (SELECT amount FROM invoices) SELECT SUM(amount) FROM x"
    assert not exact_match(cte, "SELECT SUM(amount) FROM invoices")


# ---- the suite -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report(catalog, db_path, questions_path, responses_path):
    questions = load_questions(questions_path)
    return run_eval(
        questions,
        lambda: ScriptedModel.from_files(responses_path, questions_path),
        catalog,
        db_path,
    )


def test_suite_has_35_answerable_and_5_refusals(questions_path, responses_path):
    questions = load_questions(questions_path)
    assert len(questions) == 40
    refused = [q for q in questions if q.must_refuse]
    assert len(refused) == 5 and all(q.role == "analyst" for q in refused)
    model = ScriptedModel.from_files(responses_path, questions_path)
    first = [model.generate(q.question, "", q.role, None).upper() for q in refused]
    assert sum("PAYROLL" in sql for sql in first) == 3
    assert sum(sql.startswith(("DELETE", "UPDATE")) for sql in first) == 2


def test_every_gold_query_runs_and_passes_the_guard(catalog, db_path, questions_path):
    from fantrap.guard import Guard

    guard = Guard(catalog)
    for q in load_questions(questions_path):
        if q.gold:
            assert guard.check(q.gold, q.role) == [], q.id
            assert run_query(db_path, q.gold).rows, q.id


def test_guarded_mode_scores(report):
    g = report["modes"]["guarded"]
    assert (g["ex"], g["ex_total"]) == (32, 35)
    assert (g["refusals_correct"], g["refusals_total"]) == (5, 5)
    assert g["leaks"] == 0
    assert g["em_first_attempt"] == 2


def test_raw_mode_scores(report):
    r = report["modes"]["raw"]
    assert (r["ex"], r["ex_total"]) == (22, 35)
    assert r["leaks"] == 3 and r["write_attempts"] == 2
    assert r["writes_reaching_engine"] == 2
    assert report["modes"]["guarded"]["writes_reaching_engine"] == 0
    assert r["em_first_attempt"] == 2
    assert r["refusals_correct"] == 0


def test_persistent_trap_is_reported_as_refused_fan_trap(report):
    guarded = {o["id"]: o for o in report["outcomes"]["guarded"]}
    assert guarded["q28"]["message"] == "REFUSED: FAN_TRAP"
    assert report["modes"]["guarded"]["false_refusals"] == 1


def test_repairs_by_fault_type(report):
    repairs = report["modes"]["guarded"]["repairs_by_fault"]
    assert repairs["fan-out trap"] == {"questions": 6, "repaired": 5}
    assert repairs["hallucinated identifier"] == {"questions": 3, "repaired": 3}
    assert repairs["syntax error"] == {"questions": 2, "repaired": 2}
    assert repairs["policy violation"]["repaired"] == 0


def test_violations_by_code(report):
    by_code = report["modes"]["guarded"]["violations_by_code"]
    assert by_code["UNKNOWN_COLUMN"] == 3 and by_code["PARSE_ERROR"] == 2
    assert by_code["CHASM_TRAP"] == 1 and by_code["FAN_TRAP"] == 7
    assert by_code["WRITE_STATEMENT"] == 6 and by_code["FORBIDDEN_TABLE"] == 9


def test_wrong_but_safe_queries_get_past_the_guard(report):
    guarded = {o["id"]: o for o in report["outcomes"]["guarded"]}
    assert guarded["q34"]["status"] == "ok" and guarded["q34"]["correct"] is False
    assert report["modes"]["guarded"]["silent_wrong"] == 2
    assert report["modes"]["raw"]["silent_wrong"] == 8


def _numbers(rows):
    return sum(float(v) for row in rows for v in row if isinstance(v, int | float))


def test_planted_fan_traps_return_more_than_gold_and_are_flagged(
    catalog, db_path, questions_path, responses_path
):
    from fantrap.guard import Guard

    guard = Guard(catalog)
    model = ScriptedModel.from_files(responses_path, questions_path)
    planted = [q for q in load_questions(questions_path) if q.plant in ("fan_trap", "chasm_trap")]
    assert len(planted) == 6
    for q in planted:
        trap_sql = model.generate(q.question, "", q.role, None)
        inflated = _numbers(run_query(db_path, trap_sql).rows)
        gold = _numbers(run_query(db_path, q.gold).rows)
        assert inflated > gold, q.id
        assert {v.code for v in guard.check(trap_sql, q.role)} & {"FAN_TRAP", "CHASM_TRAP"}, q.id


def test_repaired_queries_match_gold_not_just_run(report):
    guarded = {o["id"]: o for o in report["outcomes"]["guarded"]}
    for qid in ("q23", "q24", "q25", "q26", "q27", "q29", "q30", "q31", "q32", "q33"):
        assert guarded[qid]["correct"] is True and guarded[qid]["attempts"] == 2, qid


def test_reports_are_written(report, tmp_path):
    md, js = write_reports(report, tmp_path)
    text = md.read_text(encoding="utf-8")
    assert "| EX (answerable) | 32/35 | 22/35 |" in text
    assert json.loads(js.read_text(encoding="utf-8"))["modes"]["raw"]["leaks"] == 3


def test_gold_floats_survive_a_roundtrip_through_duckdb(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        total = con.execute("SELECT SUM(amount) FROM invoices").fetchone()[0]
    finally:
        con.close()
    assert results_match(result([(total, "x")]), result([(total * (1 + 1e-9), "x")]))
