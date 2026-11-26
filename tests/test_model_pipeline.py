import json
import os

import httpx
import pytest

from fantrap.model import ModelError, OpenAICompatModel, ScriptedModel, extract_sql
from fantrap.pipeline import run_pipeline


class FixedModel:
    """Returns the listed SQL strings in order, repeating the last."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.feedback_seen = []

    def generate(self, question, schema_prompt, role, feedback):
        self.feedback_seen.append(feedback)
        index = min(len(self.feedback_seen) - 1, len(self.replies) - 1)
        return self.replies[index]


def test_scripted_model_walks_through_attempts_and_repeats_the_last(responses_path, questions_path):
    model = ScriptedModel.from_files(responses_path, questions_path)
    question = "What was invoiced in each region in 2025 on invoices that have shipped?"
    first = model.generate(question, "", "analyst", None)
    second = model.generate(question, "", "analyst", "fix it")
    third = model.generate(question, "", "analyst", "fix it again")
    assert first != second and "JOIN (SELECT invoice_id" in second
    assert third == second
    assert model.generate(question, "", "analyst", None) == first  # new run starts over


def test_scripted_model_accepts_question_ids_and_rejects_unknown_text(
    responses_path, questions_path
):
    model = ScriptedModel.from_files(responses_path, questions_path)
    assert "payroll" in model.generate("q36", "", "analyst", None)
    with pytest.raises(ModelError, match="no scripted response"):
        model.generate("something nobody recorded", "", "analyst", None)


def test_extract_sql_strips_fences():
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("  SELECT 2  ") == "SELECT 2"


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compat_request_shape_and_reply_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        reply = {"choices": [{"message": {"content": "```sql\nSELECT 1\n```"}}]}
        return httpx.Response(200, json=reply)

    model = OpenAICompatModel("https://llm.test/v1/", "k", "m", client=_mock_client(handler))
    sql = model.generate("How many?", "SCHEMA", "analyst", "Problems: x")
    assert sql == "SELECT 1"
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "m" and seen["body"]["temperature"] == 0
    system, user = (m["content"] for m in seen["body"]["messages"])
    assert "SCHEMA" in system and "How many?" in user and "Problems: x" in user


def test_openai_compat_http_error_becomes_model_error():
    client = _mock_client(lambda request: httpx.Response(500, text="boom"))
    model = OpenAICompatModel("https://llm.test/v1", "k", "m", client=client)
    with pytest.raises(ModelError, match="model request failed"):
        model.generate("q", "", "analyst", None)


def test_openai_compat_reads_environment(monkeypatch):
    for name in ("FANTRAP_LLM_BASE_URL", "FANTRAP_LLM_API_KEY", "FANTRAP_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ModelError, match="FANTRAP_LLM_BASE_URL"):
        OpenAICompatModel.from_env()
    monkeypatch.setenv("FANTRAP_LLM_BASE_URL", "https://x/v1")
    monkeypatch.setenv("FANTRAP_LLM_API_KEY", "k")
    monkeypatch.setenv("FANTRAP_LLM_MODEL", "m")
    assert OpenAICompatModel.from_env().model == "m"


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("FANTRAP_LIVE") != "1", reason="set FANTRAP_LIVE=1")
def test_live_model_answers_a_simple_question(catalog, db_path):  # pragma: no cover
    run = run_pipeline(
        "How many invoices are there?", "analyst", OpenAICompatModel.from_env(), catalog, db_path
    )
    assert run.status == "ok"


def test_clean_sql_runs_without_repair(catalog, db_path):
    run = run_pipeline(
        "q", "analyst", FixedModel("SELECT COUNT(*) AS n FROM invoices"), catalog, db_path
    )
    assert run.status == "ok" and len(run.attempts) == 1
    assert run.result.rows == [(3000,)]


def test_limit_is_injected_before_execution(catalog, db_path):
    run = run_pipeline("q", "analyst", FixedModel("SELECT * FROM shipments"), catalog, db_path)
    assert len(run.result.rows) == 1000


def test_violation_is_repaired_with_feedback_naming_the_code(catalog, db_path):
    model = FixedModel("SELECT SUM(total_amount) FROM invoices", "SELECT SUM(amount) FROM invoices")
    run = run_pipeline("q", "analyst", model, catalog, db_path)
    assert run.status == "ok" and run.repaired and len(run.attempts) == 2
    assert model.feedback_seen[0] is None
    assert "UNKNOWN_COLUMN" in model.feedback_seen[1]
    assert "SELECT SUM(total_amount) FROM invoices" in model.feedback_seen[1]


def test_persistent_violation_is_refused_after_two_repairs(catalog, db_path):
    model = FixedModel("DELETE FROM invoices")
    run = run_pipeline("q", "analyst", model, catalog, db_path)
    assert run.refused and run.message == "REFUSED: WRITE_STATEMENT"
    assert len(run.attempts) == 3 and run.result is None


def test_refusal_lists_every_persistent_code(catalog, db_path):
    sql = "SELECT SUM(i.amount) FROM invoices i JOIN shipments s ON s.invoice_id = i.invoice_id"
    run = run_pipeline("q", "analyst", FixedModel(sql), catalog, db_path)
    assert run.message == "REFUSED: FAN_TRAP"


def test_engine_error_is_fed_back_and_repaired(catalog, db_path):
    # Ambiguous column: passes the guard, fails in DuckDB, and the repair fixes it.
    ambiguous = (
        "SELECT customer_id FROM invoices i JOIN customers c ON c.customer_id = i.customer_id"
    )
    model = FixedModel(ambiguous, "SELECT i.customer_id FROM invoices i LIMIT 1")
    run = run_pipeline("q", "analyst", model, catalog, db_path)
    assert run.status == "ok" and run.attempts[0].codes == ["EXECUTION_ERROR"]
    assert "EXECUTION_ERROR" in model.feedback_seen[1]


def test_unrepairable_engine_error_ends_as_failed(catalog, db_path):
    ambiguous = (
        "SELECT customer_id FROM invoices i JOIN customers c ON c.customer_id = i.customer_id"
    )
    run = run_pipeline("q", "analyst", FixedModel(ambiguous), catalog, db_path)
    assert run.status == "failed" and run.message.startswith("FAILED:")
